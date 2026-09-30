"""Recurrent PPO over complete next-goal football segments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical

from .attention import PLAYER_ACTIVE_INDEX, TensorFrame, batch_tensor_frames, tensorize
from .env import VectorFootballEnv
from .network import ActorCritic, RecurrentState


@dataclass
class _Transition:
    frame: TensorFrame
    next_frame: TensorFrame
    state: RecurrentState | None
    action: int
    old_log_probability: float
    old_value: float
    terminal_class: int = 0


@dataclass
class _Segment:
    transitions: list[_Transition]
    result: int


@dataclass
class Rollout:
    """Padded contiguous sequences and their segment-level WDL targets."""

    encoded: torch.Tensor
    own: torch.Tensor
    opponent: torch.Tensor
    context: torch.Tensor
    anchor_indices: torch.Tensor
    opponent_anchor_indices: torch.Tensor
    next_encoded: torch.Tensor
    next_own: torch.Tensor
    next_opponent: torch.Tensor
    next_context: torch.Tensor
    next_anchor_indices: torch.Tensor
    next_opponent_anchor_indices: torch.Tensor
    initial_entity_state: torch.Tensor
    initial_global_state: torch.Tensor
    valid: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminal_classes: torch.Tensor
    reward_total: float
    transition_count: int


def _select_state(state: RecurrentState, indices: list[int]) -> RecurrentState:
    index = torch.tensor(indices, dtype=torch.long, device=state.entities.device)
    return RecurrentState(
        state.entities.index_select(0, index),
        state.global_state.index_select(0, index),
    )


def _put_state(
    state: RecurrentState, indices: list[int], selected: RecurrentState
) -> RecurrentState:
    index = torch.tensor(indices, dtype=torch.long, device=state.entities.device)
    entities = state.entities.clone()
    global_state = state.global_state.clone()
    entities.index_copy_(0, index, selected.entities)
    global_state.index_copy_(0, index, selected.global_state)
    return RecurrentState(entities, global_state)


def _clear_state(state: RecurrentState, index: int) -> RecurrentState:
    entities = state.entities.clone()
    global_state = state.global_state.clone()
    entities[index].zero_()
    global_state[index].zero_()
    return RecurrentState(entities, global_state)


def _stack_padded_frames(
    chunks: list[list[_Transition]],
    length: int,
    next_frame: bool,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    fields: list[torch.Tensor] = []
    for field in range(len(TensorFrame._fields)):
        sequences = []
        for chunk in chunks:
            values = [
                (transition.next_frame if next_frame else transition.frame)[field]
                for transition in chunk
            ]
            value = torch.stack(values)
            padding = (length - len(values), *value.shape[1:])
            if padding[0]:
                value = torch.cat((value, torch.zeros(padding, dtype=value.dtype)))
            sequences.append(value)
        fields.append(torch.stack(sequences).to(device, non_blocking=True))
    return tuple(fields)


def _pack_rollout(
    segments: list[_Segment], sequence_length: int, device: torch.device
) -> Rollout:
    chunks: list[list[_Transition]] = []
    chunk_results: list[int] = []
    for segment in segments:
        for start in range(0, len(segment.transitions), sequence_length):
            chunks.append(segment.transitions[start:start + sequence_length])
            chunk_results.append(segment.result)
    if not chunks:
        raise RuntimeError("rollout contains no scored segment")

    current = _stack_padded_frames(chunks, sequence_length, False, device)
    following = _stack_padded_frames(chunks, sequence_length, True, device)
    sequence_count = len(chunks)
    valid = torch.zeros(sequence_count, sequence_length, dtype=torch.bool)
    actions = torch.zeros(sequence_count, sequence_length, dtype=torch.long)
    old_log_probabilities = torch.zeros(sequence_count, sequence_length)
    old_values = torch.zeros(sequence_count, sequence_length)
    returns = torch.zeros(sequence_count, sequence_length)
    terminal_classes = torch.zeros(sequence_count, sequence_length, dtype=torch.long)
    initial_entities = []
    initial_globals = []
    transition_count = 0
    for index, (chunk, result) in enumerate(zip(chunks, chunk_results)):
        count = len(chunk)
        transition_count += count
        valid[index, :count] = True
        actions[index, :count] = torch.tensor([item.action for item in chunk])
        old_log_probabilities[index, :count] = torch.tensor(
            [item.old_log_probability for item in chunk]
        )
        old_values[index, :count] = torch.tensor([item.old_value for item in chunk])
        returns[index, :count] = float(result)
        terminal_classes[index, :count] = torch.tensor(
            [item.terminal_class for item in chunk]
        )
        if chunk[0].state is None:
            raise RuntimeError("sequence start is missing its recurrent state")
        initial_entities.append(chunk[0].state.entities)
        initial_globals.append(chunk[0].state.global_state)

    valid = valid.to(device)
    returns = returns.to(device)
    old_values = old_values.to(device)
    return Rollout(
        *current,
        *following,
        torch.cat(initial_entities).to(device, non_blocking=True),
        torch.cat(initial_globals).to(device, non_blocking=True),
        valid,
        actions.to(device),
        old_log_probabilities.to(device),
        returns - old_values,
        returns,
        terminal_classes.to(device),
        float(sum(segment.result for segment in segments)),
        transition_count,
    )


def collect_vector_rollout(
    environment: VectorFootballEnv,
    policy: ActorCritic,
    observations: list[dict],
    steps: int,
    sequence_length: int = 32,
    progress: Callable[[str], None] | None = None,
) -> tuple[Rollout, list[dict], list[tuple[int, int]], list[tuple[int, int]]]:
    """Collect complete scored segments and retain their causal state sequence."""

    if steps <= 0 or sequence_length <= 0:
        raise ValueError("rollout steps and sequence length must be positive")
    if len(observations) != environment.count:
        raise ValueError("one observation is required per environment")
    device = next(policy.parameters()).device
    recurrent = policy.initial_state(environment.count, device)
    pending: list[list[_Transition]] = [[] for _ in range(environment.count)]
    retained: list[_Segment] = []
    completed_segments: list[tuple[int, int]] = []
    completed_matches: list[tuple[int, int]] = []
    retained_steps = simulated_steps = wave = 0
    report_interval = max(environment.count, min(512, max(steps // 8, 1)))
    next_report = report_interval
    policy.eval()

    while retained_steps < steps:
        wave += 1
        active = [True] * environment.count
        while any(active):
            decision_indices = [
                index for index in range(environment.count)
                if active[index] and observations[index]["is_in_play"]
            ]
            frames = [tensorize(observations[index]) for index in decision_indices]
            actions = [0] * environment.count
            decisions: dict[
                int, tuple[TensorFrame, RecurrentState | None, int, float, float]
            ] = {}
            if frames:
                inputs = batch_tensor_frames(frames, device)
                before = _select_state(recurrent, decision_indices)
                with torch.inference_mode():
                    distribution, output = policy.distributions(*inputs, before)
                    sampled = distribution.sample()
                    log_probability = distribution.log_prob(sampled)
                recurrent = _put_state(recurrent, decision_indices, output.state)
                for local, environment_index in enumerate(decision_indices):
                    action = int(sampled[local])
                    actions[environment_index] = action
                    sequence_start = len(pending[environment_index]) % sequence_length == 0
                    saved_state = (
                        RecurrentState(
                            before.entities[local:local + 1].cpu(),
                            before.global_state[local:local + 1].cpu(),
                        )
                        if sequence_start else None
                    )
                    decisions[environment_index] = (
                        frames[local],
                        saved_state,
                        action,
                        float(log_probability[local]),
                        float(output.value[local]),
                    )

            next_observations, _, segment_done, infos = environment.step(actions, active)
            for index in range(environment.count):
                if not active[index]:
                    continue
                if infos[index]["action_applied"]:
                    frame, before, action, log_probability, value = decisions[index]
                    pending[index].append(
                        _Transition(
                            frame, tensorize(next_observations[index]), before,
                            action, log_probability, value,
                        )
                    )
                    simulated_steps += 1
                if segment_done[index]:
                    result = infos[index]["segment_result"]
                    if result not in (-1, 0, 1):
                        raise RuntimeError("completed segment has no WDL result")
                    if not pending[index]:
                        raise RuntimeError("a completed segment has no policy decision")
                    pending[index][-1].next_frame = tensorize(next_observations[index])
                    pending[index][-1].terminal_class = (
                        1 if result == 1 else 2 if result == -1 else 0
                    )
                    completed_segments.append((int(result), infos[index]["segment_steps"]))
                    if result:
                        retained.append(_Segment(pending[index], int(result)))
                        retained_steps += len(pending[index])
                    pending[index] = []
                    recurrent = _clear_state(recurrent, index)
                    active[index] = False
                    if infos[index]["match_done"]:
                        completed_matches.append(tuple(infos[index]["goals"]))
                        next_observations[index] = environment.reset_one(index)
            observations = next_observations
            if progress is not None and simulated_steps >= next_report:
                progress(
                    f"steps={retained_steps} target={steps} simulated={simulated_steps} "
                    f"wave={wave} active={sum(active)}/{environment.count} "
                    f"segments={len(completed_segments)} matches={len(completed_matches)}"
                )
                next_report += report_interval
        if progress is not None:
            progress(
                f"wave={wave} complete steps={retained_steps} target={steps} "
                f"simulated={simulated_steps} segments={len(completed_segments)} "
                f"matches={len(completed_matches)}"
            )

    return (
        _pack_rollout(retained, sequence_length, device),
        observations,
        completed_segments,
        completed_matches,
    )


def update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 4,
    batch_size: int = 256,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.001,
    transition_coefficient: float = 0.1,
    action_value_coefficient: float = 0.1,
    control_coefficient: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Optimize PPO, EPV, and engine-grounded auxiliary predictions together."""

    valid_advantages = rollout.advantages[rollout.valid]
    normalized_advantages = torch.zeros_like(rollout.advantages)
    normalized_advantages[rollout.valid] = (
        valid_advantages - valid_advantages.mean()
    ) / (valid_advantages.std(unbiased=False) + 1e-8)
    sequence_count, sequence_length = rollout.encoded.shape[:2]
    sequences_per_batch = max(batch_size // sequence_length, 1)
    batch_count = math.ceil(sequence_count / sequences_per_batch)
    report_interval = max(batch_count // 4, 1)
    names = (
        "policy", "value", "entropy", "kl", "clip_fraction",
        "outcome", "terminal", "latent", "action_value", "control",
    )
    totals = {name: 0.0 for name in names}
    total_samples = 0
    policy.train()

    for epoch in range(1, epochs + 1):
        epoch_totals = {name: 0.0 for name in names}
        epoch_samples = 0
        batches = torch.randperm(
            sequence_count, device=rollout.encoded.device
        ).split(sequences_per_batch)
        for batch_number, indices in enumerate(batches, start=1):
            valid = rollout.valid[indices]
            output = policy.sequence(
                rollout.encoded[indices], rollout.own[indices],
                rollout.opponent[indices], rollout.context[indices],
                rollout.anchor_indices[indices], rollout.opponent_anchor_indices[indices],
                RecurrentState(
                    rollout.initial_entity_state[indices],
                    rollout.initial_global_state[indices],
                ),
                valid,
            )
            distribution = Categorical(logits=output.logits)
            log_probability = distribution.log_prob(rollout.actions[indices])
            old_log_probability = rollout.old_log_probabilities[indices]
            log_ratio = log_probability - old_log_probability
            ratio = log_ratio.exp()
            advantages = normalized_advantages[indices]
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.minimum(unclipped, clipped)[valid].mean()
            value_loss = nn.functional.mse_loss(
                output.value[valid], rollout.returns[indices][valid]
            )
            entropy = distribution.entropy()[valid].mean()
            approximate_kl = ((ratio - 1.0) - log_ratio)[valid].mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_ratio)[valid].float().mean()

            action_index = rollout.actions[indices].unsqueeze(-1)
            selected_outcome = output.predicted_outcome.gather(
                2, action_index[..., None].expand(-1, -1, 1, output.predicted_outcome.shape[-1])
            ).squeeze(2)
            selected_terminal = output.terminal_logits.gather(
                2, action_index[..., None].expand(-1, -1, 1, 3)
            ).squeeze(2)
            selected_latent = output.predicted_latent.gather(
                2, action_index[..., None].expand(-1, -1, 1, output.predicted_latent.shape[-1])
            ).squeeze(2)
            selected_action_value = output.action_values.gather(2, action_index).squeeze(-1)
            target_outcome = policy.observed_outcome(
                rollout.next_own[indices].reshape(-1, 11, rollout.next_own.shape[-1]),
                rollout.next_opponent[indices].reshape(-1, 11, rollout.next_opponent.shape[-1]),
                rollout.next_context[indices].reshape(-1, rollout.next_context.shape[-1]),
                rollout.next_anchor_indices[indices].reshape(-1),
            ).reshape(*valid.shape, -1)
            outcome_loss = nn.functional.mse_loss(selected_outcome[valid], target_outcome[valid])
            terminal_loss = nn.functional.cross_entropy(
                selected_terminal[valid], rollout.terminal_classes[indices][valid]
            )
            latent_valid = valid[:, :-1] & valid[:, 1:]
            latent_valid &= rollout.terminal_classes[indices, :-1] == 0
            if latent_valid.any():
                latent_loss = nn.functional.smooth_l1_loss(
                    selected_latent[:, :-1][latent_valid],
                    output.latent[:, 1:].detach()[latent_valid],
                )
            else:
                latent_loss = selected_latent.sum() * 0.0
            action_value_loss = nn.functional.mse_loss(
                selected_action_value[valid], rollout.returns[indices][valid]
            )

            players = torch.cat((rollout.own[indices], rollout.opponent[indices]), dim=2)
            reach_target = players[..., 20]
            reach_valid = valid[..., None] & (players[..., PLAYER_ACTIVE_INDEX] > 0.5)
            reach_valid &= reach_target > 0.0
            if reach_valid.any():
                control_loss = nn.functional.smooth_l1_loss(
                    torch.log1p(output.reach_ball[reach_valid] * 10.0),
                    torch.log1p(reach_target[reach_valid] * 10.0),
                )
            else:
                control_loss = output.reach_ball.sum() * 0.0

            loss = (
                policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
                + transition_coefficient * (outcome_loss + terminal_loss + latent_loss)
                + action_value_coefficient * action_value_loss
                + control_coefficient * control_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is NaN or Inf")
            optimizer.step()

            count = int(valid.sum())
            metrics = {
                "policy": policy_loss.item(), "value": value_loss.item(),
                "entropy": entropy.item(), "kl": approximate_kl.item(),
                "clip_fraction": clip_fraction.item(), "outcome": outcome_loss.item(),
                "terminal": terminal_loss.item(), "latent": latent_loss.item(),
                "action_value": action_value_loss.item(), "control": control_loss.item(),
            }
            for name, metric in metrics.items():
                totals[name] += metric * count
                epoch_totals[name] += metric * count
            total_samples += count
            epoch_samples += count
            if progress is not None and (
                batch_number % report_interval == 0 or batch_number == batch_count
            ):
                progress(f"epoch={epoch}/{epochs} batch={batch_number}/{batch_count}")
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} complete "
                f"policy={epoch_totals['policy'] / epoch_samples:.4f} "
                f"value={epoch_totals['value'] / epoch_samples:.4f} "
                f"transition={(epoch_totals['outcome'] + epoch_totals['terminal'] + epoch_totals['latent']) / epoch_samples:.4f} "
                f"q={epoch_totals['action_value'] / epoch_samples:.4f} "
                f"entropy={epoch_totals['entropy'] / epoch_samples:.4f}"
            )
    return {name: value / total_samples for name, value in totals.items()}
