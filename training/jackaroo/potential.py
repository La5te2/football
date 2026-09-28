"""Outcome-supervised ODA-conditioned learned potential and replay training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from .attention import ODAEncoder, batch_histories
from .dataset import Episode, attention_targets, mirror_observation


Observation = dict[str, Any]
@dataclass
class _Record:
    episode: Episode
    index: int
    mirrored: bool
    outcome: int
    offense: int
    defense: int
    offense_weight: float
    defense_weight: float
    stratum: tuple[int, int, int]


def _record_history(
    record: _Record, history_length: int, opposite: bool = False
) -> list[Observation]:
    start = max(0, record.index - history_length + 1)
    history = record.episode.observations[start : record.index + 1]
    if record.mirrored != opposite:
        return [mirror_observation(observation) for observation in history]
    return history


class ODAConditionedPotential(nn.Module):
    """Predict W/D/L value and ODA targets from public observations."""

    def __init__(self, frame_size: int) -> None:
        super().__init__()
        self.oda = ODAEncoder()
        self.global_encoder = nn.Sequential(
            nn.Linear(frame_size, 128),
            nn.Tanh(),
        )
        self.temporal = nn.GRU(128 + self.oda.output_width, 128, batch_first=True)
        self.outcome_head = nn.Linear(128, 3)

    def forward(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation, offense_logits, defense_logits = self.oda(
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        sequence = torch.cat((self.global_encoder(encoded), relation), dim=-1)
        temporal, _ = self.temporal(sequence)
        outcome_logits = self.outcome_head(temporal[:, -1])
        probabilities = torch.softmax(outcome_logits, dim=-1)
        potential = probabilities[:, 0] - probabilities[:, 2]
        return (
            potential,
            outcome_logits,
            offense_logits[:, -1],
            defense_logits[:, -1],
        )

    @torch.no_grad()
    def predict(
        self,
        observation_histories: Sequence[Sequence[Observation]],
        maximum_steps: int,
        history_length: int,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        encoded, own, opponent, context, anchors, opponent_anchors = batch_histories(
            observation_histories, maximum_steps, history_length, device
        )
        result = self(
            encoded, own, opponent, context, anchors, opponent_anchors
        )[0]
        if was_training:
            self.train()
        return result


class PotentialEnsemble(nn.Module):
    """Independent potential members with disagreement-aware prediction."""

    def __init__(
        self, frame_size: int, member_count: int = 3, uncertainty_scale: float = 5.0
    ) -> None:
        super().__init__()
        if member_count <= 0:
            raise ValueError("potential ensemble must contain at least one member")
        if uncertainty_scale < 0.0:
            raise ValueError("potential uncertainty scale must be nonnegative")
        self.members = nn.ModuleList(
            ODAConditionedPotential(frame_size) for _ in range(member_count)
        )
        self.uncertainty_scale = uncertainty_scale

    def forward(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = [
            member(
                encoded,
                own,
                opponent,
                context,
                anchor_indices,
                opponent_anchor_indices,
            )
            for member in self.members
        ]
        probabilities = torch.stack(
            [torch.softmax(output[1], dim=-1) for output in outputs]
        ).mean(dim=0)
        logits = probabilities.clamp_min(1e-8).log()
        potential = probabilities[:, 0] - probabilities[:, 2]
        offense = torch.stack([output[2] for output in outputs]).mean(dim=0)
        defense = torch.stack([output[3] for output in outputs]).mean(dim=0)
        return potential, logits, offense, defense

    @torch.no_grad()
    def predict(
        self,
        observation_histories: Sequence[Sequence[Observation]],
        maximum_steps: int,
        history_length: int,
    ) -> torch.Tensor:
        predictions = torch.stack(
            [
                member.predict(
                    observation_histories, maximum_steps, history_length
                )
                for member in self.members
            ]
        )
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0, unbiased=False)
        confidence = torch.exp(-self.uncertainty_scale * variance)
        return confidence * mean

    def averaged_oda(self) -> ODAEncoder:
        """Average pretrained ODA parameters for policy initialization."""

        result = copy.deepcopy(self.members[0].oda)
        states = [member.oda.state_dict() for member in self.members]
        averaged = {
            name: torch.stack([state[name] for state in states]).mean(dim=0)
            if states[0][name].is_floating_point()
            else states[0][name]
            for name in states[0]
        }
        result.load_state_dict(averaged)
        return result


class FrozenPotential:
    """Inference-only snapshot used for one complete PPO interval."""

    def __init__(self, model: PotentialEnsemble) -> None:
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(
        self,
        observation_history: Sequence[Observation],
        maximum_steps: int,
        history_length: int,
    ) -> float:
        return float(
            self.model.predict(
                [observation_history], maximum_steps, history_length
            )[0].item()
        )


def potential_change(
    previous: PotentialEnsemble,
    candidate: PotentialEnsemble,
    episodes: Sequence[Episode],
    maximum_steps: int,
    history_length: int,
    batch_size: int = 256,
) -> float:
    """Return mean absolute potential change on complete validation matches."""

    histories = [
        episode.observations[max(0, index - history_length + 1) : index + 1]
        for episode in episodes
        for index in range(len(episode.observations) - 1)
    ]
    if not histories:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, len(histories), batch_size):
        selected = histories[start : start + batch_size]
        first = previous.predict(selected, maximum_steps, history_length)
        second = candidate.predict(selected, maximum_steps, history_length)
        total += (second - first).abs().sum().item()
        count += len(selected)
    return total / count


def _records(
    episodes: Sequence[Episode], history_length: int
) -> list[_Record]:
    records: list[_Record] = []
    for episode in episodes:
        if len(episode.observations) < 2:
            continue
        off, deff, off_weight, def_weight = attention_targets(episode.observations)
        final_step = max(episode.observations[-1]["step"], 1)
        for index, observation in enumerate(episode.observations[:-1]):
            time_bucket = min(2, observation["step"] * 3 // final_step)
            difference = observation["goals"][0] - observation["goals"][1]
            score_bucket = 0 if difference < 0 else 1 if difference == 0 else 2
            records.append(
                _Record(
                    episode,
                    index,
                    False,
                    episode.outcome,
                    off[index],
                    deff[index],
                    off_weight[index],
                    def_weight[index],
                    (episode.outcome, time_bucket, score_bucket),
                )
            )
            records.append(
                _Record(
                    episode,
                    index,
                    True,
                    2 - episode.outcome,
                    deff[index],
                    off[index],
                    def_weight[index],
                    off_weight[index],
                    (2 - episode.outcome, time_bucket, 2 - score_bucket),
                )
            )
    return records


def _weighted_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    valid = targets >= 0
    if not valid.any():
        return torch.zeros((), device=logits.device)
    losses = nn.functional.cross_entropy(
        logits[valid], targets[valid], reduction="none"
    )
    return (losses * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)


def fit_potential(
    model: ODAConditionedPotential | PotentialEnsemble,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Episode],
    maximum_steps: int,
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 128,
    oda_weight: float = 0.25,
    symmetry_weight: float = 0.1,
    history_length: int = 4,
) -> dict[str, float]:
    """Fit outcome, confidence-weighted ODA, and mirror-consistency losses."""

    if isinstance(model, PotentialEnsemble):
        results = [
            fit_potential(
                member,
                optimizer,
                episodes,
                maximum_steps,
                device,
                epochs,
                batch_size,
                oda_weight,
                symmetry_weight,
                history_length,
            )
            for member in model.members
        ]
        return {
            name: sum(result[name] for result in results) / len(results)
            for name in results[0]
        } if results else {"outcome": 0.0, "oda": 0.0, "symmetry": 0.0}

    records = _records(episodes, history_length)
    if not records:
        return {"outcome": 0.0, "oda": 0.0, "symmetry": 0.0}

    model.train()
    totals = {"outcome": 0.0, "oda": 0.0, "symmetry": 0.0}
    updates = 0
    stratum_counts: dict[tuple[int, int, int], int] = {}
    for record in records:
        stratum_counts[record.stratum] = stratum_counts.get(record.stratum, 0) + 1
    weights = torch.tensor(
        [1.0 / stratum_counts[record.stratum] for record in records]
    )
    for _ in range(epochs):
        order = torch.multinomial(weights, len(records), replacement=True)
        for indices in order.split(batch_size):
            selected = [records[index] for index in indices.tolist()]
            original_histories = [
                _record_history(record, history_length) for record in selected
            ]
            mirror_histories = [
                _record_history(record, history_length, opposite=True)
                for record in selected
            ]
            encoded, own, opponent, context, anchors, opponent_anchors = batch_histories(
                original_histories, maximum_steps, history_length, device
            )
            (
                mirror_encoded,
                mirror_own,
                mirror_opponent,
                mirror_context,
                mirror_anchors,
                mirror_opponent_anchors,
            ) = batch_histories(
                mirror_histories, maximum_steps, history_length, device
            )
            potential, outcome_logits, offense_logits, defense_logits = model(
                encoded,
                own,
                opponent,
                context,
                anchors,
                opponent_anchors,
            )
            mirror_potential = model(
                mirror_encoded,
                mirror_own,
                mirror_opponent,
                mirror_context,
                mirror_anchors,
                mirror_opponent_anchors,
            )[0]
            outcome_target = torch.tensor(
                [record.outcome for record in selected], dtype=torch.long, device=device
            )
            offense_target = torch.tensor(
                [record.offense for record in selected], dtype=torch.long, device=device
            )
            defense_target = torch.tensor(
                [record.defense for record in selected], dtype=torch.long, device=device
            )
            offense_weight = torch.tensor(
                [record.offense_weight for record in selected],
                dtype=torch.float32,
                device=device,
            )
            defense_weight = torch.tensor(
                [record.defense_weight for record in selected],
                dtype=torch.float32,
                device=device,
            )
            outcome_loss = nn.functional.cross_entropy(outcome_logits, outcome_target)
            oda_loss = _weighted_cross_entropy(
                offense_logits, offense_target, offense_weight
            ) + _weighted_cross_entropy(defense_logits, defense_target, defense_weight)
            symmetry_loss = nn.functional.mse_loss(mirror_potential, -potential)
            loss = outcome_loss + oda_weight * oda_loss + symmetry_weight * symmetry_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("potential loss is NaN or Inf")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("potential gradient norm is NaN or Inf")
            optimizer.step()
            totals["outcome"] += outcome_loss.item()
            totals["oda"] += oda_loss.item()
            totals["symmetry"] += symmetry_loss.item()
            updates += 1
    return {name: value / max(updates, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_potential(
    model: ODAConditionedPotential | PotentialEnsemble,
    episodes: Sequence[Episode],
    maximum_steps: int,
    device: torch.device,
    history_length: int = 4,
    batch_size: int = 256,
) -> dict[str, float]:
    """Measure outcome calibration, ODA accuracy, entropy, and symmetry."""

    records = _records(episodes, history_length)
    if not records:
        return {}
    was_training = model.training
    model.eval()
    outcome_count = 0
    outcome_correct = 0
    outcome_cross_entropy = 0.0
    brier = 0.0
    symmetry = 0.0
    confidence_bins = [0.0] * 10
    accuracy_bins = [0.0] * 10
    count_bins = [0] * 10
    grouped: dict[tuple[int, int], list[float]] = {}
    oda_totals = {
        "offense_count": 0.0,
        "offense_top1": 0.0,
        "offense_top3": 0.0,
        "offense_entropy": 0.0,
        "offense_cross_entropy": 0.0,
        "defense_count": 0.0,
        "defense_top1": 0.0,
        "defense_top3": 0.0,
        "defense_entropy": 0.0,
        "defense_cross_entropy": 0.0,
    }
    oda_bins = {
        prefix: [[0.0, 0.0, 0.0] for _ in range(10)]
        for prefix in ("offense", "defense")
    }
    for start in range(0, len(records), batch_size):
        selected = records[start : start + batch_size]
        histories = [_record_history(record, history_length) for record in selected]
        mirror_histories = [
            _record_history(record, history_length, opposite=True)
            for record in selected
        ]
        inputs = batch_histories(
            histories, maximum_steps, history_length, device
        )
        mirror_inputs = batch_histories(
            mirror_histories, maximum_steps, history_length, device
        )
        potential, logits, offense_logits, defense_logits = model(*inputs)
        mirror_potential = model(*mirror_inputs)[0]
        targets = torch.tensor(
            [record.outcome for record in selected], dtype=torch.long, device=device
        )
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        outcome_count += len(selected)
        outcome_correct += (predictions == targets).sum().item()
        outcome_cross_entropy += nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        one_hot = nn.functional.one_hot(targets, 3).float()
        brier += ((probabilities - one_hot) ** 2).sum(dim=-1).sum().item()
        symmetry += ((mirror_potential + potential) ** 2).sum().item()
        confidence, predicted = probabilities.max(dim=-1)
        for value, correct in zip(
            confidence.tolist(), (predicted == targets).tolist()
        ):
            index = min(int(value * 10), 9)
            confidence_bins[index] += value
            accuracy_bins[index] += float(correct)
            count_bins[index] += 1
        for record, probability, correct in zip(
            selected, confidence.tolist(), (predicted == targets).tolist()
        ):
            key = (record.stratum[1], record.stratum[2])
            summary = grouped.setdefault(key, [0.0, 0.0, 0.0])
            summary[0] += 1.0
            summary[1] += float(correct)
            summary[2] += probability

        for prefix, branch_logits, target_values, weight_values in (
            (
                "offense",
                offense_logits,
                [record.offense for record in selected],
                [record.offense_weight for record in selected],
            ),
            (
                "defense",
                defense_logits,
                [record.defense for record in selected],
                [record.defense_weight for record in selected],
            ),
        ):
            branch_targets = torch.tensor(
                target_values, dtype=torch.long, device=device
            )
            branch_weights = torch.tensor(
                weight_values, dtype=torch.float32, device=device
            )
            valid = (branch_targets >= 0) & (branch_weights > 0)
            if valid.any():
                branch_probabilities = torch.softmax(branch_logits[valid], dim=-1)
                valid_targets = branch_targets[valid]
                valid_weights = branch_weights[valid]
                weight_sum = valid_weights.sum().item()
                oda_totals[f"{prefix}_count"] += weight_sum
                oda_totals[f"{prefix}_top1"] += (
                    (branch_probabilities.argmax(dim=-1) == valid_targets).float()
                    * valid_weights
                ).sum().item()
                oda_totals[f"{prefix}_top3"] += (
                    (branch_probabilities.topk(3, dim=-1).indices ==
                     valid_targets.unsqueeze(-1)).any(dim=-1).float()
                    * valid_weights
                ).sum().item()
                entropy = -(
                    branch_probabilities
                    * branch_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1)
                oda_totals[f"{prefix}_entropy"] += (
                    entropy * valid_weights
                ).sum().item()
                cross_entropy = nn.functional.cross_entropy(
                    branch_logits[valid], valid_targets, reduction="none"
                )
                oda_totals[f"{prefix}_cross_entropy"] += (
                    cross_entropy * valid_weights
                ).sum().item()
                branch_confidence, branch_prediction = branch_probabilities.max(
                    dim=-1
                )
                for confidence, correct, weight in zip(
                    branch_confidence.tolist(),
                    (branch_prediction == valid_targets).tolist(),
                    valid_weights.tolist(),
                ):
                    index = min(int(confidence * 10), 9)
                    oda_bins[prefix][index][0] += weight
                    oda_bins[prefix][index][1] += confidence * weight
                    oda_bins[prefix][index][2] += float(correct) * weight

    ece = 0.0
    for confidence, accuracy, count in zip(
        confidence_bins, accuracy_bins, count_bins
    ):
        if count:
            ece += count / outcome_count * abs(accuracy / count - confidence / count)
    metrics = {
        "outcome_accuracy": outcome_correct / outcome_count,
        "outcome_cross_entropy": outcome_cross_entropy / outcome_count,
        "brier": brier / outcome_count,
        "calibration_error": ece,
        "symmetry": symmetry / outcome_count,
    }
    for prefix in ("offense", "defense"):
        count = max(oda_totals[f"{prefix}_count"], 1e-8)
        metrics[f"{prefix}_top1"] = oda_totals[f"{prefix}_top1"] / count
        metrics[f"{prefix}_top3"] = oda_totals[f"{prefix}_top3"] / count
        metrics[f"{prefix}_entropy"] = oda_totals[f"{prefix}_entropy"] / count
        metrics[f"{prefix}_cross_entropy"] = (
            oda_totals[f"{prefix}_cross_entropy"] / count
        )
        metrics[f"{prefix}_calibration_error"] = sum(
            weight / count * abs(correct / weight - confidence / weight)
            for weight, confidence, correct in oda_bins[prefix]
            if weight > 0.0
        )
    for (time_bucket, score_bucket), (count, correct, confidence) in grouped.items():
        prefix = f"time_{time_bucket}_score_{score_bucket}"
        metrics[f"{prefix}_accuracy"] = correct / count
        metrics[f"{prefix}_confidence"] = confidence / count
    if was_training:
        model.train()
    return metrics


def make_potential_function(
    snapshot: FrozenPotential, maximum_steps: int, history_length: int
) -> Callable[[Sequence[Observation]], float]:
    """Adapt a frozen model snapshot to the environment reward callback."""

    return lambda history: snapshot(history, maximum_steps, history_length)
