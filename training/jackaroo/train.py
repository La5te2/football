"""Train Jackaroo from scored next-goal segments with recurrent PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .env import FootballEnv, VectorFootballEnv
from .evaluate import evaluate_policy
from .features import encode
from .network import ActorCritic, POLICY_ARCHITECTURE, POLICY_OBJECTIVE
from .ppo import collect_vector_rollout, update


def _status(message: str) -> None:
    print(f"status={message}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--transition-coefficient", type=float, default=0.1)
    parser.add_argument("--action-value-coefficient", type=float, default=0.1)
    parser.add_argument("--control-coefficient", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jackaroo.pt"))
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--evaluation-games", type=int, default=2)
    return parser.parse_args()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        "updates", "steps_per_update", "environments", "maximum_steps",
        "sequence_length", "ppo_epochs", "ppo_batch_size", "evaluation_interval",
    )
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if arguments.evaluation_games < 0:
        raise ValueError("evaluation-games must be nonnegative")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    nonnegative = (
        "entropy_coefficient", "transition_coefficient",
        "action_value_coefficient", "control_coefficient",
    )
    for name in nonnegative:
        if getattr(arguments, name) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    log_path = arguments.log or arguments.checkpoint.with_suffix(
        arguments.checkpoint.suffix + ".jsonl"
    )
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
    _status(
        f"startup device={device.type} updates={arguments.updates} "
        f"environments={arguments.environments} seed={arguments.seed}"
    )

    bootstrap_environment = FootballEnv(arguments.maximum_steps, arguments.seed)
    observation = bootstrap_environment.reset(arguments.seed)
    frame_size = encode(observation).numel()
    action_count = bootstrap_environment.action_count
    policy = ActorCritic(frame_size, action_count).to(device)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    _status(f"policy architecture={POLICY_ARCHITECTURE} parameters={parameter_count}")
    start_iteration = 1
    checkpoint: dict[str, Any] = {}
    if arguments.resume is not None:
        _status(f"resume loading={arguments.resume}")
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=True)
        expected = {
            "architecture": POLICY_ARCHITECTURE,
            "parameter_count": parameter_count,
            "frame_size": frame_size,
            "action_count": action_count,
            "maximum_steps": arguments.maximum_steps,
            "sequence_length": arguments.sequence_length,
            "objective": POLICY_OBJECTIVE,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint.get(name)!r} does not match "
                    f"requested value {value!r}"
                )
        policy.load_state_dict(checkpoint["model"])
        start_iteration = int(checkpoint.get("updates", 0)) + 1
    del bootstrap_environment

    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    environment = VectorFootballEnv(
        arguments.environments,
        arguments.maximum_steps,
        arguments.seed,
        fixed_seed=arguments.seed,
    )
    observations = environment.reset()
    _status(
        f"vector-environment count={arguments.environments} fixed_seed={arguments.seed} ready"
    )

    for iteration in range(start_iteration, start_iteration + arguments.updates):
        _status(f"update={iteration} rollout start target_steps={arguments.steps_per_update}")
        rollout, observations, segments, matches = collect_vector_rollout(
            environment,
            policy,
            observations,
            arguments.steps_per_update,
            sequence_length=arguments.sequence_length,
            progress=lambda message: _status(f"update={iteration} rollout {message}"),
        )
        _status(f"update={iteration} ppo start")
        losses = update(
            policy,
            optimizer,
            rollout,
            epochs=arguments.ppo_epochs,
            batch_size=arguments.ppo_batch_size,
            entropy_coefficient=arguments.entropy_coefficient,
            transition_coefficient=arguments.transition_coefficient,
            action_value_coefficient=arguments.action_value_coefficient,
            control_coefficient=arguments.control_coefficient,
            progress=lambda message: _status(f"update={iteration} ppo {message}"),
        )
        action_counts = torch.bincount(
            rollout.actions[rollout.valid].detach().cpu(), minlength=action_count
        ).float()
        segment_wins = sum(result == 1 for result, _ in segments)
        censored_segments = sum(result == 0 for result, _ in segments)
        segment_losses = sum(result == -1 for result, _ in segments)
        retained_segments = [segment for segment in segments if segment[0] != 0]
        mean_segment_steps = sum(length for _, length in retained_segments) / max(
            len(retained_segments), 1
        )
        diagnostics = {
            "action_distribution": (action_counts / action_counts.sum()).tolist(),
            "reward_total": rollout.reward_total,
            "transition_count": rollout.transition_count,
            "segment_wins": segment_wins,
            "segment_losses": segment_losses,
            "censored_segments": censored_segments,
            "mean_segment_steps": mean_segment_steps,
        }
        evaluation: dict[str, float] = {}
        if arguments.evaluation_games and iteration % arguments.evaluation_interval == 0:
            _status(f"update={iteration} evaluation start")
            evaluation = evaluate_policy(
                policy,
                arguments.maximum_steps,
                arguments.evaluation_games,
                arguments.seed + 1_000_000,
                progress=lambda message: _status(f"update={iteration} evaluation {message}"),
            )
        scores = " ".join(f"{left}:{right}" for left, right in matches) or "-"
        print(
            f"update={iteration} matches={scores} "
            f"segments={segment_wins}/{segment_losses} censored={censored_segments} "
            f"segment_steps={mean_segment_steps:.1f} policy={losses['policy']:.4f} "
            f"value={losses['value']:.4f} entropy={losses['entropy']:.4f} "
            f"transition={losses['outcome'] + losses['terminal'] + losses['latent']:.4f} "
            f"q={losses['action_value']:.4f} control={losses['control']:.4f} "
            f"kl={losses['kl']:.5f} clipped={losses['clip_fraction']:.4f} "
            f"reward={rollout.reward_total:.3f}",
            flush=True,
        )
        checkpoint = {
            "architecture": POLICY_ARCHITECTURE,
            "parameter_count": parameter_count,
            "model": policy.state_dict(),
            "frame_size": frame_size,
            "action_count": action_count,
            "maximum_steps": arguments.maximum_steps,
            "sequence_length": arguments.sequence_length,
            "environments": arguments.environments,
            "seed": arguments.seed,
            "ppo_epochs": arguments.ppo_epochs,
            "ppo_batch_size": arguments.ppo_batch_size,
            "entropy_coefficient": arguments.entropy_coefficient,
            "transition_coefficient": arguments.transition_coefficient,
            "action_value_coefficient": arguments.action_value_coefficient,
            "control_coefficient": arguments.control_coefficient,
            "objective": POLICY_OBJECTIVE,
            "feature_scaling": "fixed",
            "updates": iteration,
            "diagnostics": diagnostics,
            "evaluation": evaluation,
            "optimizer": optimizer.state_dict(),
        }
        _status(f"update={iteration} checkpoint saving")
        _atomic_torch_save(checkpoint, arguments.checkpoint)
        _append_log(
            log_path,
            {
                "update": iteration,
                "segments": segments,
                "matches": matches,
                "ppo": losses,
                "diagnostics": diagnostics,
                "evaluation": evaluation,
            },
        )
        _status(f"update={iteration} checkpoint saved={arguments.checkpoint}")


if __name__ == "__main__":
    main()
