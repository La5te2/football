"""Train Jackaroo from scored next-goal segments with recurrent PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from .env import FootballEnv, SelfPlayVectorFootballEnv
from .evaluate import evaluate_policy
from .features import encode
from .network import ActorCritic, POLICY_ARCHITECTURE, POLICY_OBJECTIVE
from .ppo import collect_self_play_rollout, update


def _status(message: str) -> None:
    print(f"status={message}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--games-per-update", type=int, default=256)
    parser.add_argument("--flight", type=int, default=16)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--transition-coefficient", type=float, default=0.1)
    parser.add_argument("--action-value-coefficient", type=float, default=0.1)
    parser.add_argument("--control-coefficient", type=float, default=0.05)
    parser.add_argument("--space-coefficient", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/jackaroo.pt"))
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument(
        "--validation-directory", type=Path, default=Path("runs/validation")
    )
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
        "flight", "maximum_steps",
        "sequence_length", "ppo_epochs", "ppo_batch_size",
    )
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    nonnegative_integers = ("updates", "games_per_update")
    for name in nonnegative_integers:
        if getattr(arguments, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
    if arguments.games_per_update == 0 and arguments.updates > 0:
        raise ValueError("games-per-update must be positive when updates are requested")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    nonnegative = (
        "entropy_coefficient", "transition_coefficient",
        "action_value_coefficient", "control_coefficient",
        "space_coefficient",
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
        f"flight={arguments.flight} games_per_update={arguments.games_per_update} "
        f"seed={arguments.seed}"
    )

    if arguments.resume is not None:
        metadata = torch.load(arguments.resume, map_location="cpu", weights_only=True)
        frame_size = int(metadata["frame_size"])
        action_count = int(metadata["action_count"])
    else:
        bootstrap_environment = FootballEnv(arguments.maximum_steps, arguments.seed)
        observation = bootstrap_environment.reset(arguments.seed)
        frame_size = encode(observation).numel()
        action_count = bootstrap_environment.action_count
        del bootstrap_environment
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
            "seed": arguments.seed,
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    sampled_matches = int(checkpoint.get("sampled_matches", 0))
    pretraining_metrics = checkpoint.get("pretraining", {})

    if arguments.updates == 0:
        _status("training complete updates=0")
        return

    environment = SelfPlayVectorFootballEnv(
        arguments.flight,
        arguments.maximum_steps,
        arguments.seed,
    )
    environment.advance_seed_sequence(sampled_matches)
    _status(
        f"self-play flight={arguments.flight} seed_sequence={arguments.seed} ready"
    )

    for iteration in range(start_iteration, start_iteration + arguments.updates):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _status(
            f"update={iteration} rollout start "
            f"target_games={arguments.games_per_update}"
        )
        rollout_started = time.perf_counter()
        rollout, segments, matches = collect_self_play_rollout(
            environment,
            policy,
            arguments.games_per_update,
            sequence_length=arguments.sequence_length,
            progress=lambda message: _status(f"update={iteration} rollout {message}"),
        )
        rollout_seconds = time.perf_counter() - rollout_started
        sampled_matches += len(matches)
        _status(f"update={iteration} ppo start")
        ppo_started = time.perf_counter()
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
            space_coefficient=arguments.space_coefficient,
            progress=lambda message: _status(f"update={iteration} ppo {message}"),
        )
        ppo_seconds = time.perf_counter() - ppo_started
        action_counts = torch.bincount(
            rollout.actions[rollout.valid].detach().cpu(), minlength=action_count
        ).float()
        left_scoring_segments = sum(result == 1 for result, _ in segments)
        censored_segments = sum(result == 0 for result, _ in segments)
        right_scoring_segments = sum(result == -1 for result, _ in segments)
        retained_segments = [segment for segment in segments if segment[0] != 0]
        mean_segment_steps = sum(length for _, length in retained_segments) / max(
            len(retained_segments), 1
        )
        engine_steps = sum(length for _, length in segments)
        diagnostics = {
            "action_distribution": (action_counts / action_counts.sum()).tolist(),
            "transition_count": rollout.transition_count,
            "left_scoring_segments": left_scoring_segments,
            "right_scoring_segments": right_scoring_segments,
            "policy_segments": 2 * (
                left_scoring_segments + right_scoring_segments
            ),
            "censored_segments": censored_segments,
            "mean_segment_steps": mean_segment_steps,
            "rollout_seconds": rollout_seconds,
            "engine_steps_per_second": engine_steps / max(rollout_seconds, 1e-9),
            "ppo_seconds": ppo_seconds,
            "ppo_transitions_per_second": (
                rollout.transition_count / max(ppo_seconds, 1e-9)
            ),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else 0
            ),
        }
        _status(f"update={iteration} validation seed={arguments.validation_seed} start")
        evaluation = evaluate_policy(
            policy,
            arguments.maximum_steps,
            2,
            arguments.validation_seed,
            progress=lambda message: _status(f"update={iteration} validation {message}"),
            record_path=(
                arguments.validation_directory / f"update-{iteration:06d}.gfr"
            ),
        )
        total_left_goals = sum(left for left, _ in matches)
        total_right_goals = sum(right for _, right in matches)
        validation_scores = " ".join(
            f"{left}:{right}" for left, right in evaluation["scores"]
        )
        print(
            f"update={iteration} self_play_matches={len(matches)} "
            f"self_play_goals={total_left_goals}:{total_right_goals} "
            f"goal_segments={left_scoring_segments}:{right_scoring_segments} "
            f"policy_segments={2 * (left_scoring_segments + right_scoring_segments)} "
            f"tails={censored_segments} "
            f"segment_steps={mean_segment_steps:.1f} policy={losses['policy']:.4f} "
            f"value={losses['value']:.4f} entropy={losses['entropy']:.4f} "
            f"transition={losses['outcome'] + losses['terminal'] + losses['latent']:.4f} "
            f"q={losses['action_value']:.4f} control={losses['control']:.4f} "
            f"space={losses['space']:.4f} "
            f"kl={losses['kl']:.5f} clipped={losses['clip_fraction']:.4f} "
            f"engine_steps_s={diagnostics['engine_steps_per_second']:.1f} "
            f"ppo_transitions_s={diagnostics['ppo_transitions_per_second']:.1f} "
            f"gpu_peak_mib={diagnostics['peak_gpu_memory_bytes'] / 1048576.0:.1f} "
            f"validation={validation_scores}",
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
            "flight": arguments.flight,
            "games_per_update": arguments.games_per_update,
            "sampled_matches": sampled_matches,
            "seed": arguments.seed,
            "ppo_epochs": arguments.ppo_epochs,
            "ppo_batch_size": arguments.ppo_batch_size,
            "entropy_coefficient": arguments.entropy_coefficient,
            "transition_coefficient": arguments.transition_coefficient,
            "action_value_coefficient": arguments.action_value_coefficient,
            "control_coefficient": arguments.control_coefficient,
            "space_coefficient": arguments.space_coefficient,
            "objective": POLICY_OBJECTIVE,
            "feature_scaling": "fixed",
            "updates": iteration,
            "pretraining": pretraining_metrics,
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
