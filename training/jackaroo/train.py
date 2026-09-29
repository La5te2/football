"""Behavior-cloning bootstrap followed by PPO fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .attention import batch_tensor_histories, tensorize
from .env import FootballEnv, POTENTIAL_DISCOUNT, VectorFootballEnv
from .features import encode
from .imitation import collect_demonstrations, pretrain_policy
from .network import ActorCritic
from .ppo import collect_vector_rollout, update


def _status(message: str) -> None:
    print(f"status={message}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/jackaroo.pt")
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--imitation-games", type=int, default=8)
    parser.add_argument("--imitation-epochs", type=int, default=3)
    parser.add_argument("--imitation-batch-size", type=int, default=256)
    parser.add_argument("--imitation-learning-rate", type=float, default=3e-4)
    parser.add_argument("--potential-scale", type=float, default=0.20)
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
    ):
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("imitation_games", "imitation_epochs", "evaluation_games"):
        if getattr(arguments, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
    if arguments.learning_rate <= 0.0 or arguments.imitation_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if arguments.potential_scale < 0.0:
        raise ValueError("potential-scale must be nonnegative")


def _evaluate(
    policy: ActorCritic,
    maximum_steps: int,
    history_length: int,
    games: int,
    seed: int,
) -> dict[str, float]:
    if games == 0:
        return {"wins": 0.0, "draws": 0.0, "losses": 0.0, "goal_difference": 0.0}
    device = next(policy.parameters()).device
    environment = FootballEnv(maximum_steps, seed)
    wins = draws = losses = goal_difference = 0
    policy.eval()
    for game in range(games):
        observation = environment.reset((seed + game // 2) & 0xFFFFFFFF)
        history = []
        terminated = False
        info: dict[str, Any] = {"goals": (0, 0)}
        while not terminated:
            history.append(tensorize(observation, maximum_steps))
            if len(history) > history_length:
                del history[:-history_length]
            with torch.inference_mode():
                logits, _ = policy(
                    *batch_tensor_histories([history], history_length, device)
                )
            observation, _, terminated, info = environment.step(
                logits[0].argmax().item()
            )
        own, opponent = info["goals"]
        wins += own > opponent
        draws += own == opponent
        losses += own < opponent
        goal_difference += own - opponent
    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "goal_difference": goal_difference / games,
    }


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
    frame_size = encode(observation, arguments.maximum_steps).numel()
    policy = ActorCritic(frame_size, bootstrap_environment.action_count).to(device)
    imitation_metrics = {"loss": 0.0, "accuracy": 0.0, "samples": 0.0}
    start_iteration = 1

    if arguments.resume is not None:
        _status(f"resume loading={arguments.resume}")
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=True)
        expected = {
            "frame_size": frame_size,
            "action_count": bootstrap_environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "reward": "goal-result-fixed-potential",
            "potential_scale": arguments.potential_scale,
            "potential_discount": POTENTIAL_DISCOUNT,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint.get(name)!r} does not match "
                    f"requested value {value!r}"
                )
        policy.load_state_dict(checkpoint["model"])
        imitation_metrics = checkpoint.get("imitation", imitation_metrics)
        start_iteration = int(checkpoint.get("updates", 0)) + 1
    elif arguments.imitation_games > 0 and arguments.imitation_epochs > 0:
        _status(f"imitation collect games={arguments.imitation_games}")
        imitation_environment = VectorFootballEnv(
            min(arguments.environments, arguments.imitation_games),
            arguments.maximum_steps,
            arguments.seed,
        )
        episodes = collect_demonstrations(
            imitation_environment,
            arguments.imitation_games,
            arguments.seed,
            progress=lambda message: _status(f"imitation {message}"),
        )
        _status("imitation optimize start")
        imitation_metrics = pretrain_policy(
            policy,
            episodes,
            arguments.history_length,
            device,
            epochs=arguments.imitation_epochs,
            batch_size=arguments.imitation_batch_size,
            learning_rate=arguments.imitation_learning_rate,
            progress=lambda message: _status(f"imitation {message}"),
        )
        del episodes
        _status(
            f"imitation complete samples={int(imitation_metrics['samples'])} "
            f"accuracy={imitation_metrics['accuracy']:.4f}"
        )
        del imitation_environment
    del bootstrap_environment

    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    if arguments.resume is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    environment = VectorFootballEnv(
        arguments.environments, arguments.maximum_steps, arguments.seed + 1
    )
    environment.set_potential_scale(arguments.potential_scale)
    observations = environment.reset()
    policy_histories = [[] for _ in range(arguments.environments)]
    _status(f"vector-environment count={arguments.environments} ready")

    for iteration in range(start_iteration, start_iteration + arguments.updates):
        _status(f"update={iteration} rollout start target_steps={arguments.steps_per_update}")
        rollout, observations, games = collect_vector_rollout(
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
            progress=lambda message: _status(f"update={iteration} ppo {message}"),
        )
        action_counts = torch.bincount(
            rollout.actions.detach().cpu(), minlength=environment.action_count
        ).float()
        diagnostics = {
            "action_distribution": (action_counts / action_counts.sum()).tolist(),
            "task_reward_total": rollout.task_reward_total,
            "potential_reward_total": rollout.potential_reward_total,
            "potential_telescoping_error_total": rollout.telescoping_error_total,
        }
        evaluation: dict[str, float] = {}
        if arguments.evaluation_games and iteration % arguments.evaluation_interval == 0:
            _status(f"update={iteration} evaluation start")
            evaluation = _evaluate(
                policy,
                arguments.maximum_steps,
                arguments.history_length,
                arguments.evaluation_games,
                arguments.seed + 1_000_000,
            )
            _status(
                f"update={iteration} evaluation "
                f"wdl={int(evaluation['wins'])}/"
                f"{int(evaluation['draws'])}/{int(evaluation['losses'])} "
                f"goal_difference={evaluation['goal_difference']:.3f}"
            )
        scores = " ".join(f"{left}:{right}" for left, right in games) or "-"
        print(
            f"update={iteration} games={scores} policy={losses['policy']:.4f} "
            f"value={losses['value']:.4f} entropy={losses['entropy']:.4f} "
            f"task={rollout.task_reward_total:.3f} "
            f"shaping={rollout.potential_reward_total:.3f}",
            flush=True,
        )
        checkpoint = {
            "model": policy.state_dict(),
            "frame_size": frame_size,
            "action_count": environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "environments": arguments.environments,
            "potential_scale": arguments.potential_scale,
            "potential_discount": POTENTIAL_DISCOUNT,
            "reward": "goal-result-fixed-potential",
            "feature_scaling": "fixed",
            "updates": iteration,
            "imitation": imitation_metrics,
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
                "games": games,
                "ppo": losses,
                "diagnostics": diagnostics,
                "evaluation": evaluation,
            },
        )
        _status(f"update={iteration} checkpoint saved={arguments.checkpoint}")


if __name__ == "__main__":
    main()
