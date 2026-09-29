"""Behavior-cloning bootstrap followed by PPO fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .env import FootballEnv, VectorFootballEnv
from .evaluate import evaluate_policy
from .features import encode
from .imitation import collect_demonstrations, pretrain_policy
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/jackaroo.pt")
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--imitation-games", type=int, default=128)
    parser.add_argument("--imitation-epochs", type=int, default=2)
    parser.add_argument("--imitation-batch-size", type=int, default=256)
    parser.add_argument("--imitation-learning-rate", type=float, default=3e-4)
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
    for name in (
        "updates",
        "steps_per_update",
        "environments",
        "maximum_steps",
        "history_length",
        "imitation_batch_size",
        "evaluation_interval",
        "ppo_epochs",
        "ppo_batch_size",
    ):
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("imitation_games", "imitation_epochs", "evaluation_games"):
        if getattr(arguments, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
    if arguments.learning_rate <= 0.0 or arguments.imitation_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if arguments.entropy_coefficient < 0.0:
        raise ValueError("entropy-coefficient must be nonnegative")


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
        f"environments={arguments.environments} maximum_steps={arguments.maximum_steps}"
    )

    bootstrap_environment = FootballEnv(arguments.maximum_steps, arguments.seed)
    observation = bootstrap_environment.reset(arguments.seed)
    frame_size = encode(observation).numel()
    policy = ActorCritic(frame_size, bootstrap_environment.action_count).to(device)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    _status(
        f"policy architecture={POLICY_ARCHITECTURE} parameters={parameter_count}"
    )
    imitation_metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "accuracy": 0.0,
        "samples": 0.0,
        "matches": 0.0,
    }
    baseline_evaluation: dict[str, float] = {}
    start_iteration = 1

    if arguments.resume is not None:
        _status(f"resume loading={arguments.resume}")
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=True)
        expected = {
            "architecture": POLICY_ARCHITECTURE,
            "parameter_count": parameter_count,
            "frame_size": frame_size,
            "action_count": bootstrap_environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "objective": POLICY_OBJECTIVE,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint.get(name)!r} does not match "
                    f"requested value {value!r}"
                )
        policy.load_state_dict(checkpoint["model"])
        imitation_metrics = checkpoint.get("imitation", imitation_metrics)
        baseline_evaluation = checkpoint.get("baseline_evaluation", {})
        start_iteration = int(checkpoint.get("updates", 0)) + 1
    elif arguments.imitation_games > 0 and arguments.imitation_epochs > 0:
        _status(f"imitation collect games={arguments.imitation_games}")
        imitation_environment = VectorFootballEnv(
            min(arguments.environments, arguments.imitation_games),
            arguments.maximum_steps,
            arguments.seed,
        )
        demonstrations = collect_demonstrations(
            imitation_environment,
            arguments.imitation_games,
            arguments.seed,
            progress=lambda message: _status(f"imitation {message}"),
        )
        _status("imitation optimize start")
        imitation_metrics = pretrain_policy(
            policy,
            demonstrations,
            arguments.history_length,
            device,
            epochs=arguments.imitation_epochs,
            batch_size=arguments.imitation_batch_size,
            learning_rate=arguments.imitation_learning_rate,
            progress=lambda message: _status(f"imitation {message}"),
        )
        del demonstrations
        _status(
            f"imitation complete samples={int(imitation_metrics['samples'])} "
            f"matches={int(imitation_metrics['matches'])} "
            f"accuracy={imitation_metrics['accuracy']:.4f}"
        )
        del imitation_environment
        if arguments.evaluation_games:
            _status("imitation baseline-evaluation start")
            baseline_evaluation = evaluate_policy(
                policy,
                arguments.maximum_steps,
                arguments.history_length,
                arguments.evaluation_games,
                arguments.seed + 1_000_000,
            )
            _status(
                "imitation baseline-evaluation "
                f"matches={int(baseline_evaluation['wins'])}/"
                f"{int(baseline_evaluation['draws'])}/"
                f"{int(baseline_evaluation['losses'])} "
                f"goal_difference={baseline_evaluation['goal_difference']:.3f} "
                f"segments={int(baseline_evaluation['segment_wins'])}/"
                f"{int(baseline_evaluation['segment_draws'])}/"
                f"{int(baseline_evaluation['segment_losses'])}"
            )
    del bootstrap_environment

    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    if arguments.resume is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    environment = VectorFootballEnv(
        arguments.environments, arguments.maximum_steps, arguments.seed + 1
    )
    observations = environment.reset()
    policy_histories = [[] for _ in range(arguments.environments)]
    _status(f"vector-environment count={arguments.environments} ready")

    for iteration in range(start_iteration, start_iteration + arguments.updates):
        _status(f"update={iteration} rollout start target_steps={arguments.steps_per_update}")
        rollout, observations, segments, matches = collect_vector_rollout(
            environment,
            policy,
            observations,
            arguments.steps_per_update,
            policy_histories=policy_histories,
            history_length=arguments.history_length,
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
            progress=lambda message: _status(f"update={iteration} ppo {message}"),
        )
        action_counts = torch.bincount(
            rollout.actions.detach().cpu(), minlength=environment.action_count
        ).float()
        segment_wins = sum(result == 1 for result, _ in segments)
        censored_segments = sum(result == 0 for result, _ in segments)
        segment_losses = sum(result == -1 for result, _ in segments)
        retained_segments = [segment for segment in segments if segment[0] != 0]
        diagnostics = {
            "action_distribution": (action_counts / action_counts.sum()).tolist(),
            "reward_total": rollout.reward_total,
            "segment_wins": segment_wins,
            "segment_losses": segment_losses,
            "censored_segments": censored_segments,
            "mean_segment_steps": (
                sum(length for _, length in retained_segments)
                / len(retained_segments)
            ),
        }
        evaluation: dict[str, float] = {}
        if arguments.evaluation_games and iteration % arguments.evaluation_interval == 0:
            _status(f"update={iteration} evaluation start")
            evaluation = evaluate_policy(
                policy,
                arguments.maximum_steps,
                arguments.history_length,
                arguments.evaluation_games,
                arguments.seed + 1_000_000,
            )
            _status(
                f"update={iteration} evaluation "
                f"matches={int(evaluation['wins'])}/"
                f"{int(evaluation['draws'])}/{int(evaluation['losses'])} "
                f"goal_difference={evaluation['goal_difference']:.3f} "
                f"segments={int(evaluation['segment_wins'])}/"
                f"{int(evaluation['segment_draws'])}/"
                f"{int(evaluation['segment_losses'])} "
                f"segment_steps={evaluation['mean_segment_steps']:.1f}"
            )
        scores = " ".join(f"{left}:{right}" for left, right in matches) or "-"
        print(
            f"update={iteration} matches={scores} "
            f"segments={segment_wins}/{segment_losses} "
            f"censored={censored_segments} "
            f"segment_steps={diagnostics['mean_segment_steps']:.1f} "
            f"policy={losses['policy']:.4f} "
            f"value={losses['value']:.4f} entropy={losses['entropy']:.4f} "
            f"kl={losses['kl']:.5f} clipped={losses['clip_fraction']:.4f} "
            f"reward={rollout.reward_total:.3f}",
            flush=True,
        )
        checkpoint = {
            "architecture": POLICY_ARCHITECTURE,
            "parameter_count": parameter_count,
            "model": policy.state_dict(),
            "frame_size": frame_size,
            "action_count": environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "environments": arguments.environments,
            "ppo_epochs": arguments.ppo_epochs,
            "ppo_batch_size": arguments.ppo_batch_size,
            "entropy_coefficient": arguments.entropy_coefficient,
            "objective": POLICY_OBJECTIVE,
            "feature_scaling": "fixed",
            "updates": iteration,
            "imitation": imitation_metrics,
            "baseline_evaluation": baseline_evaluation,
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
