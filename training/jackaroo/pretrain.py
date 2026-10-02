"""GPU pretraining from a prepared Jackaroo trajectory dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterator

import h5py
import numpy as np
import torch
from torch import nn

from .attention import OPPONENT_POSSESSION_INDEX, OWN_POSSESSION_INDEX, PLAYER_ACTIVE_INDEX
from .network import (
    ActorCritic,
    ENTITY_COUNT,
    ENTITY_STATE_WIDTH,
    GLOBAL_STATE_WIDTH,
    POLICY_ARCHITECTURE,
    POLICY_OBJECTIVE,
    RecurrentState,
)


class DatasetReader:
    """Read prepared trajectory batches directly into GPU pretraining."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = h5py.File(path, "r")
        self.sequence_length = int(self._file.attrs["sequence_length"])
        self.sequence_count = int(self._file.attrs["sequences"])
        self.frame_size = int(self._file["encoded"].shape[-1])
        self.transitions = int(self._file.attrs["transitions"])
        self.matches = int(self._file.attrs["matches"])
        self.scored_segments = int(self._file.attrs["scored_segments"])
        self.censored_segments = int(self._file.attrs["censored_segments"])
        self.trajectory_count = int(self._file.attrs["trajectories"])
        self.trajectory_ids = self._file["trajectory"][:].astype(np.int64)
        self.trajectory_ends = self._file["trajectory_end"][:].astype(np.bool_)
        if self.sequence_count <= 0:
            raise ValueError("prepared dataset contains no sequences")
        if self.trajectory_count <= 0:
            raise ValueError("prepared dataset contains no trajectories")

    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        """Yield shuffled trajectories one ordered truncated sequence at a time."""

        sequences_per_batch = max(batch_size // self.sequence_length, 1)
        grouped: list[list[int]] = [[] for _ in range(self.trajectory_count)]
        for sequence, trajectory in enumerate(self.trajectory_ids):
            grouped[int(trajectory)].append(sequence)
        order = torch.randperm(self.trajectory_count).tolist()
        cursor = 0
        active: list[tuple[list[int], int]] = []
        while cursor < len(order) and len(active) < sequences_per_batch:
            active.append((grouped[order[cursor]], 0))
            cursor += 1
        while active:
            yield np.asarray(
                [trajectory[position] for trajectory, position in active],
                dtype=np.int64,
            )
            following: list[tuple[list[int], int]] = []
            for trajectory, position in active:
                if position + 1 < len(trajectory):
                    following.append((trajectory, position + 1))
                elif cursor < len(order):
                    following.append((grouped[order[cursor]], 0))
                    cursor += 1
            active = following

    def read(
        self, name: str, indices: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        order = np.argsort(indices)
        value = self._file[name][indices[order]]
        value = value[np.argsort(order)]
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        return tensor.to(device, non_blocking=True)

    def initial_state(
        self, count: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(count, ENTITY_COUNT, ENTITY_STATE_WIDTH, device=device),
            torch.zeros(count, GLOBAL_STATE_WIDTH, device=device),
        )

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> "DatasetReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def pretrain(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    dataset: DatasetReader,
    epochs: int = 2,
    batch_size: int = 256,
    value_coefficient: float = 0.5,
    transition_coefficient: float = 0.1,
    action_value_coefficient: float = 0.1,
    control_coefficient: float = 0.05,
    space_coefficient: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Initialize Jackaroo from compressed teacher trajectories."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("pretraining epochs and batch size must be positive")
    names = (
        "behavior", "value", "outcome", "terminal", "latent",
        "action_value", "control", "space", "accuracy",
    )
    totals = {name: 0.0 for name in names}
    weights = {name: 0 for name in names}
    device = next(policy.parameters()).device
    policy.train()

    for epoch in range(1, epochs + 1):
        epoch_totals = {name: 0.0 for name in names}
        epoch_weights = {name: 0 for name in names}
        recurrent: dict[int, RecurrentState] = {}
        batches = list(dataset.batches(batch_size))
        batch_count = len(batches)
        report_interval = max(batch_count // 4, 1)
        for batch_number, indices in enumerate(batches, start=1):
            read = lambda name: dataset.read(name, indices, device)
            encoded = read("encoded")
            own = read("own")
            opponent = read("opponent")
            context = read("context")
            anchor_indices = read("anchor").long()
            opponent_anchor_indices = read("opponent_anchor").long()
            valid = read("valid").bool()
            policy_valid = read("policy_valid").bool()
            terminal_supervision = read("terminal_supervision").bool()
            action_supervision = read("action_supervision").bool()
            actions = read("actions").long()
            returns = read("returns").float()
            terminal_classes = read("terminal_classes").long()
            next_own = read("next_own")
            next_opponent = read("next_opponent")
            next_context = read("next_context")
            next_anchor_indices = read("next_anchor").long()
            initial_entities, initial_global = dataset.initial_state(len(indices), device)
            trajectory_ids = dataset.trajectory_ids[indices]
            for row, trajectory in enumerate(trajectory_ids):
                state = recurrent.get(int(trajectory))
                if state is not None:
                    initial_entities[row] = state.entities
                    initial_global[row] = state.global_state

            supervised = valid & action_supervision
            value_supervised = policy_valid
            terminal_supervised = supervised & terminal_supervision
            action_value_supervised = policy_valid & action_supervision
            output = policy.sequence(
                encoded, own, opponent, context,
                anchor_indices, opponent_anchor_indices,
                RecurrentState(initial_entities, initial_global), valid,
            )
            for row, trajectory in enumerate(trajectory_ids):
                key = int(trajectory)
                if dataset.trajectory_ends[indices[row]]:
                    recurrent.pop(key, None)
                else:
                    recurrent[key] = RecurrentState(
                        output.state.entities[row].detach(),
                        output.state.global_state[row].detach(),
                    )
            action_index = actions.unsqueeze(-1)
            selected_outcome = output.predicted_outcome.gather(
                2,
                action_index[..., None].expand(
                    -1, -1, 1, output.predicted_outcome.shape[-1]
                ),
            ).squeeze(2)
            selected_terminal = output.terminal_logits.gather(
                2, action_index[..., None].expand(-1, -1, 1, 3)
            ).squeeze(2)
            selected_latent = output.predicted_latent.gather(
                2,
                action_index[..., None].expand(
                    -1, -1, 1, output.predicted_latent.shape[-1]
                ),
            ).squeeze(2)
            selected_action_value = output.action_values.gather(
                2, action_index
            ).squeeze(-1)

            if supervised.any():
                behavior_loss = nn.functional.cross_entropy(
                    output.logits[supervised], actions[supervised]
                )
                accuracy = (
                    output.logits[supervised].argmax(dim=-1)
                    == actions[supervised]
                ).float().mean()
                target_outcome = policy.observed_outcome(
                    next_own.reshape(-1, 11, next_own.shape[-1]),
                    next_opponent.reshape(-1, 11, next_opponent.shape[-1]),
                    next_context.reshape(-1, next_context.shape[-1]),
                    next_anchor_indices.reshape(-1),
                ).reshape(*valid.shape, -1)
                outcome_loss = nn.functional.mse_loss(
                    selected_outcome[supervised], target_outcome[supervised]
                )
            else:
                zero = output.logits.sum() * 0.0
                behavior_loss = accuracy = outcome_loss = zero

            if value_supervised.any():
                value_loss = nn.functional.mse_loss(
                    output.value[value_supervised], returns[value_supervised]
                )
            else:
                value_loss = output.value.sum() * 0.0
            if terminal_supervised.any():
                terminal_loss = nn.functional.cross_entropy(
                    selected_terminal[terminal_supervised],
                    terminal_classes[terminal_supervised],
                )
            else:
                terminal_loss = selected_terminal.sum() * 0.0
            if action_value_supervised.any():
                action_value_loss = nn.functional.mse_loss(
                    selected_action_value[action_value_supervised],
                    returns[action_value_supervised],
                )
            else:
                action_value_loss = selected_action_value.sum() * 0.0

            latent_valid = supervised[:, :-1] & valid[:, 1:]
            latent_valid &= terminal_classes[:, :-1] == 0
            if latent_valid.any():
                latent_loss = nn.functional.smooth_l1_loss(
                    selected_latent[:, :-1][latent_valid],
                    output.latent[:, 1:].detach()[latent_valid],
                )
            else:
                latent_loss = selected_latent.sum() * 0.0

            players = torch.cat((own, opponent), dim=2)
            reach_target = players[..., 20]
            reach_valid = valid[..., None] & (
                players[..., PLAYER_ACTIVE_INDEX] > 0.5
            )
            reach_valid &= reach_target > 0.0
            if reach_valid.any():
                control_loss = nn.functional.smooth_l1_loss(
                    torch.log1p(output.reach_ball[reach_valid] * 10.0),
                    torch.log1p(reach_target[reach_valid] * 10.0),
                )
            else:
                control_loss = output.reach_ball.sum() * 0.0

            space_target = policy.control.analytic_space_value_target(
                own.reshape(-1, 11, own.shape[-1]),
                opponent.reshape(-1, 11, opponent.shape[-1]),
            ).reshape(*valid.shape, 2, -1)
            possession = torch.stack(
                (
                    context[..., OWN_POSSESSION_INDEX] > 0.5,
                    context[..., OPPONENT_POSSESSION_INDEX] > 0.5,
                ),
                dim=-1,
            )
            space_valid = (valid[..., None] & possession)[..., None].expand_as(
                output.space_values
            )
            if space_valid.any():
                space_loss = nn.functional.smooth_l1_loss(
                    output.space_values[space_valid], space_target[space_valid]
                )
            else:
                space_loss = output.space_values.sum() * 0.0

            loss = (
                behavior_loss
                + value_coefficient * value_loss
                + transition_coefficient * (
                    outcome_loss + terminal_loss + latent_loss
                )
                + action_value_coefficient * action_value_loss
                + control_coefficient * control_loss
                + space_coefficient * space_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("pretraining loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("pretraining gradient norm is NaN or Inf")
            optimizer.step()

            metric_weights = {
                "behavior": int(supervised.sum()),
                "value": int(value_supervised.sum()),
                "outcome": int(supervised.sum()),
                "terminal": int(terminal_supervised.sum()),
                "latent": int(latent_valid.sum()),
                "action_value": int(action_value_supervised.sum()),
                "control": int(reach_valid.sum()),
                "space": int(space_valid.sum()),
                "accuracy": int(supervised.sum()),
            }
            metrics = {
                "behavior": behavior_loss.item(), "value": value_loss.item(),
                "outcome": outcome_loss.item(), "terminal": terminal_loss.item(),
                "latent": latent_loss.item(),
                "action_value": action_value_loss.item(),
                "control": control_loss.item(), "space": space_loss.item(),
                "accuracy": accuracy.item(),
            }
            for name, metric in metrics.items():
                weight = metric_weights[name]
                totals[name] += metric * weight
                weights[name] += weight
                epoch_totals[name] += metric * weight
                epoch_weights[name] += weight
            if progress is not None and (
                batch_number % report_interval == 0 or batch_number == batch_count
            ):
                progress(f"epoch={epoch}/{epochs} batch={batch_number}/{batch_count}")
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} complete "
                f"behavior={epoch_totals['behavior'] / max(epoch_weights['behavior'], 1):.4f} "
                f"value={epoch_totals['value'] / max(epoch_weights['value'], 1):.4f} "
                f"accuracy={epoch_totals['accuracy'] / max(epoch_weights['accuracy'], 1):.4f}"
            )
    return {name: totals[name] / max(weights[name], 1) for name in names}


def _status(message: str) -> None:
    print(f"status={message}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("runs/tamakeri.h5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jackaroo.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--transition-coefficient", type=float, default=0.1)
    parser.add_argument("--action-value-coefficient", type=float, default=0.1)
    parser.add_argument("--control-coefficient", type=float, default=0.05)
    parser.add_argument("--space-coefficient", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if arguments.maximum_steps <= 0 or arguments.learning_rate <= 0.0:
        raise ValueError("maximum steps and learning rate must be positive")
    for name in (
        "transition_coefficient", "action_value_coefficient",
        "control_coefficient", "space_coefficient",
    ):
        if getattr(arguments, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    with DatasetReader(arguments.data) as dataset:
        policy = ActorCritic(dataset.frame_size, 32).to(device)
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=arguments.learning_rate
        )
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        _status(
            f"pretraining start device={device.type} data={arguments.data} "
            f"matches={dataset.matches} transitions={dataset.transitions} "
            f"sequences={dataset.sequence_count} epochs={arguments.epochs}"
        )
        losses = pretrain(
            policy,
            optimizer,
            dataset,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            transition_coefficient=arguments.transition_coefficient,
            action_value_coefficient=arguments.action_value_coefficient,
            control_coefficient=arguments.control_coefficient,
            space_coefficient=arguments.space_coefficient,
            progress=lambda message: _status(f"pretraining {message}"),
        )
        pretraining_metrics = {
            "teacher": "tamakeri",
            "dataset": str(arguments.data),
            "games": dataset.matches,
            "scored_segments": dataset.scored_segments,
            "censored_segments": dataset.censored_segments,
            "transitions": dataset.transitions,
            "losses": losses,
        }
        checkpoint = {
            "architecture": POLICY_ARCHITECTURE,
            "parameter_count": parameter_count,
            "model": policy.state_dict(),
            "frame_size": dataset.frame_size,
            "action_count": 32,
            "maximum_steps": arguments.maximum_steps,
            "sequence_length": dataset.sequence_length,
            "sampled_matches": 0,
            "seed": arguments.seed,
            "objective": POLICY_OBJECTIVE,
            "feature_scaling": "fixed",
            "updates": 0,
            "pretraining": pretraining_metrics,
            "diagnostics": {},
            "evaluation": {},
            "optimizer": optimizer.state_dict(),
        }
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.checkpoint.with_name(arguments.checkpoint.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(arguments.checkpoint)
    _status(
        f"pretraining complete accuracy={losses['accuracy']:.4f} "
        f"checkpoint={arguments.checkpoint}"
    )


if __name__ == "__main__":
    main()
