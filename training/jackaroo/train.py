"""Train a single-agent football policy with PPO."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from .dataset import Episode, EpisodeDataset, ODA_HORIZON, ODA_STABLE_STEPS
from .env import FootballEnv, VectorFootballEnv
from .features import encode
from .network import ActorCritic
from .potential import (
    FrozenPotential,
    PotentialEnsemble,
    evaluate_potential,
    fit_potential,
    make_batched_potential_function,
    potential_change,
)
from .ppo import collect_vector_rollout, update


def _status(message: str) -> None:
    """Write one immediately visible progress line for terminals and nohup logs."""

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
        "--checkpoint", type=Path, default=Path("checkpoints/jackaroo.pt")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--resume", type=Path,
        help="Resume policy, potential, and optimizer state from a checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--potential-epochs", type=int, default=2)
    parser.add_argument("--potential-buffer-size", type=int, default=64)
    parser.add_argument("--potential-learning-rate", type=float, default=1e-3)
    parser.add_argument("--potential-ensemble-size", type=int, default=3)
    parser.add_argument("--potential-uncertainty-scale", type=float, default=5.0)
    parser.add_argument("--potential-scale", type=float, default=0.02)
    parser.add_argument("--minimum-potential-scale", type=float, default=0.0025)
    parser.add_argument("--potential-bootstrap-games", type=int, default=4)
    parser.add_argument("--potential-refresh-interval", type=int, default=1)
    parser.add_argument("--potential-batches-per-epoch", type=int, default=64)
    parser.add_argument("--potential-refresh-alpha", type=float, default=0.25)
    parser.add_argument("--potential-validation-tolerance", type=float, default=0.02)
    parser.add_argument("--potential-change-limit", type=float, default=0.10)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument(
        "--training-group",
        choices=("terminal", "learned", "oda"),
        default="oda",
    )
    return parser.parse_args()


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(path.suffix + suffix)


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Replace one PyTorch artifact only after the new file is complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_log(path: Path, record: dict[str, Any]) -> None:
    """Append one bounded training record without retaining prior records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        "updates",
        "steps_per_update",
        "environments",
        "maximum_steps",
        "potential_epochs",
        "potential_buffer_size",
        "potential_ensemble_size",
        "potential_refresh_interval",
        "potential_batches_per_epoch",
        "history_length",
    )
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if arguments.potential_bootstrap_games < 0:
        raise ValueError("potential-bootstrap-games must be nonnegative")
    if arguments.learning_rate <= 0.0 or arguments.potential_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if not 0.0 < arguments.potential_refresh_alpha <= 1.0:
        raise ValueError("potential-refresh-alpha must be in (0, 1]")
    if arguments.minimum_potential_scale < 0.0:
        raise ValueError("minimum-potential-scale must be nonnegative")
    if arguments.potential_scale < arguments.minimum_potential_scale:
        raise ValueError("potential-scale must cover minimum-potential-scale")
    if arguments.potential_validation_tolerance < 0.0:
        raise ValueError("potential-validation-tolerance must be nonnegative")
    if arguments.potential_change_limit < 0.0:
        raise ValueError("potential-change-limit must be nonnegative")


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    output_dataset = arguments.dataset or _sidecar(
        arguments.checkpoint, ".episodes.pt"
    )
    log_path = arguments.log or _sidecar(arguments.checkpoint, ".jsonl")
    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)
    _status(
        f"startup device={device.type} group={arguments.training_group} "
        f"updates={arguments.updates} environments={arguments.environments} "
        f"maximum_steps={arguments.maximum_steps}"
    )
    bootstrap_environment = FootballEnv(arguments.maximum_steps, arguments.seed)
    potential_scale = arguments.potential_scale
    if arguments.training_group == "terminal":
        potential_scale = 0.0
    bootstrap_environment.set_potential_scale(potential_scale)
    observation = bootstrap_environment.reset(arguments.seed)
    frame_size = encode(observation, arguments.maximum_steps).numel()
    policy = ActorCritic(frame_size, bootstrap_environment.action_count).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)
    potential_model = PotentialEnsemble(
        frame_size,
        arguments.potential_ensemble_size,
        arguments.potential_uncertainty_scale,
    ).to(device)
    potential_optimizer = torch.optim.Adam(
        potential_model.parameters(), lr=arguments.potential_learning_rate
    )
    potential_dataset = EpisodeDataset(arguments.potential_buffer_size)
    potential_metrics = {"outcome": 0.0, "oda": 0.0, "symmetry": 0.0}
    bootstrap_episodes: list[Episode] = []
    snapshot: FrozenPotential | None = None

    start_iteration = 1
    if arguments.resume is not None:
        _status(f"resume loading={arguments.resume}")
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=True)
        expected = {
            "frame_size": frame_size,
            "action_count": bootstrap_environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "potential_ensemble_size": arguments.potential_ensemble_size,
            "feature_scaling": "fixed",
            "oda_horizon": ODA_HORIZON,
            "oda_stable_steps": ODA_STABLE_STEPS,
            "training_group": arguments.training_group,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={checkpoint.get(name)!r} does not match "
                    f"requested value {value!r}"
                )
        policy.load_state_dict(checkpoint["model"])
        potential_model.load_state_dict(checkpoint["potential_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "potential_optimizer" in checkpoint:
            potential_optimizer.load_state_dict(checkpoint["potential_optimizer"])
        potential_metrics = checkpoint.get("potential_metrics", potential_metrics)
        potential_scale = float(checkpoint.get("potential_scale", potential_scale))
        snapshot = FrozenPotential(potential_model)
        input_dataset = arguments.dataset or _sidecar(
            arguments.resume, ".episodes.pt"
        )
        if not input_dataset.is_file():
            raise FileNotFoundError(
                f"resume dataset does not exist: {input_dataset}"
            )
        potential_dataset.load_state_dict(
            torch.load(input_dataset, map_location="cpu", weights_only=True)
        )
        if len(potential_dataset) != int(checkpoint.get("potential_updates", -1)):
            raise ValueError("checkpoint and trajectory dataset are inconsistent")
        start_iteration = int(checkpoint.get("updates", 0)) + 1
        _status(
            f"resume ready start_update={start_iteration} "
            f"episodes={len(potential_dataset)}"
        )
    else:
        bootstrap_games = (
            0 if arguments.training_group == "terminal"
            else arguments.potential_bootstrap_games
        )
        for game in range(bootstrap_games):
            game_seed = arguments.seed + game
            _status(
                f"bootstrap game={game + 1}/{bootstrap_games} seed={game_seed} start"
            )
            observations, goals = bootstrap_environment.collect_builtin_episode(
                game_seed,
                progress=lambda step, maximum, index=game + 1: _status(
                    f"bootstrap game={index}/{bootstrap_games} "
                    f"step={step}/{maximum}"
                ),
            )
            left_goals, right_goals = goals
            outcome = 0 if left_goals > right_goals else 1 if left_goals == right_goals else 2
            bootstrap_episodes.append(Episode(observations, outcome))
            _status(
                f"bootstrap game={game + 1}/{bootstrap_games} "
                f"complete score={left_goals}:{right_goals}"
            )
        potential_dataset.add(bootstrap_episodes, base=True)
    if arguments.resume is None and bootstrap_episodes:
        _status(
            f"initial-potential episodes={len(potential_dataset.train)} fitting"
        )
        potential_metrics = fit_potential(
            potential_model,
            potential_optimizer,
            potential_dataset.train,
            arguments.maximum_steps,
            device,
            epochs=arguments.potential_epochs,
            oda_weight=0.0 if arguments.training_group == "learned" else 0.25,
            history_length=arguments.history_length,
            progress=lambda message: _status(f"initial-potential {message}"),
            maximum_batches_per_epoch=arguments.potential_batches_per_epoch,
        )
        if arguments.training_group == "oda":
            policy.initialize_oda(potential_model.averaged_oda())
        snapshot = FrozenPotential(potential_model)
        _status("initial-potential validation")
        validation_metrics = evaluate_potential(
            potential_model,
            potential_dataset.validation,
            arguments.maximum_steps,
            device,
            history_length=arguments.history_length,
        )
        potential_metrics.update(
            {f"validation_{name}": value for name, value in validation_metrics.items()}
        )
        _status("initial-potential test")
        test_metrics = evaluate_potential(
            potential_model,
            potential_dataset.test,
            arguments.maximum_steps,
            device,
            history_length=arguments.history_length,
        )
        potential_metrics.update(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )
        _status("initial-potential complete")
    del bootstrap_environment
    environment = VectorFootballEnv(
        arguments.environments,
        arguments.maximum_steps,
        arguments.seed + len(bootstrap_episodes),
    )
    environment.set_potential_scale(potential_scale)
    if snapshot is not None:
        environment.set_potential(
            make_batched_potential_function(
                snapshot, arguments.maximum_steps, arguments.history_length
            )
        )
    observations = environment.reset()
    episode_histories = [[] for _ in range(arguments.environments)]
    policy_histories = [[] for _ in range(arguments.environments)]
    _status(f"vector-environment count={arguments.environments} ready")

    for iteration in range(start_iteration, start_iteration + arguments.updates):
        _status(
            f"update={iteration} rollout start target_steps={arguments.steps_per_update}"
        )
        rollout, observations, games, episodes = collect_vector_rollout(
            environment,
            policy,
            observations,
            arguments.steps_per_update,
            episode_histories=episode_histories,
            policy_histories=policy_histories,
            history_length=arguments.history_length,
            progress=lambda message: _status(f"update={iteration} rollout {message}"),
        )
        _status(
            f"update={iteration} rollout complete steps={rollout.actions.numel()} "
            f"games={len(games)}"
        )
        _status(f"update={iteration} ppo start")
        losses = update(
            policy,
            optimizer,
            rollout,
            progress=lambda message: _status(f"update={iteration} ppo {message}"),
        )
        _status(f"update={iteration} ppo complete")
        action_counts = torch.bincount(
            rollout.actions.detach().cpu(), minlength=environment.action_count
        ).float()
        action_distribution = (action_counts / action_counts.sum()).tolist()
        possession_transitions = sum(
            previous["ball_owned_team"] != current["ball_owned_team"]
            and previous["ball_owned_team"] in (0, 1)
            and current["ball_owned_team"] in (0, 1)
            for episode in episodes
            for previous, current in zip(
                episode.observations, episode.observations[1:]
            )
        )
        diagnostics = {
            "action_distribution": action_distribution,
            "shots": int(action_counts[12].item()),
            "possession_transitions": possession_transitions,
            "task_reward_total": rollout.task_reward_total,
            "potential_reward_total": rollout.potential_reward_total,
            "potential_telescoping_error_total": rollout.telescoping_error_total,
        }
        potential_dataset.add(episodes)
        if (
            episodes
            and arguments.training_group != "terminal"
            and potential_dataset.train
            and potential_dataset.validation
            and iteration % arguments.potential_refresh_interval == 0
        ):
            _status(f"update={iteration} potential previous-validation")
            previous_model = copy.deepcopy(potential_model).eval()
            previous_potential = copy.deepcopy(potential_model.state_dict())
            previous_validation = evaluate_potential(
                potential_model,
                potential_dataset.validation,
                arguments.maximum_steps,
                device,
                history_length=arguments.history_length,
            )
            potential_metrics = fit_potential(
                potential_model,
                potential_optimizer,
                potential_dataset.train,
                arguments.maximum_steps,
                device,
                epochs=arguments.potential_epochs,
                oda_weight=0.0 if arguments.training_group == "learned" else 0.25,
                history_length=arguments.history_length,
                progress=lambda message: _status(
                    f"update={iteration} potential {message}"
                ),
                maximum_batches_per_epoch=arguments.potential_batches_per_epoch,
            )
            _status(f"update={iteration} potential candidate-validation")
            validation_metrics = evaluate_potential(
                potential_model,
                potential_dataset.validation,
                arguments.maximum_steps,
                device,
                history_length=arguments.history_length,
            )
            previous_loss = previous_validation.get("outcome_cross_entropy")
            candidate_loss = validation_metrics.get("outcome_cross_entropy")
            candidate_change = potential_change(
                previous_model,
                potential_model,
                potential_dataset.validation,
                arguments.maximum_steps,
                arguments.history_length,
            )
            accepted = (
                previous_loss is None
                or candidate_loss is None
                or candidate_loss
                <= previous_loss + arguments.potential_validation_tolerance
            ) and candidate_change <= arguments.potential_change_limit
            if accepted:
                alpha = arguments.potential_refresh_alpha
                if not 0.0 < alpha <= 1.0:
                    raise ValueError("potential refresh alpha must be in (0, 1]")
                candidate = potential_model.state_dict()
                blended = {
                    name: previous_potential[name].lerp(candidate[name], alpha)
                    if candidate[name].is_floating_point()
                    else candidate[name]
                    for name in candidate
                }
                potential_model.load_state_dict(blended)
                validation_metrics = evaluate_potential(
                    potential_model,
                    potential_dataset.validation,
                    arguments.maximum_steps,
                    device,
                    history_length=arguments.history_length,
                )
                _status(f"update={iteration} potential refresh=accepted")
            else:
                potential_model.load_state_dict(previous_potential)
                validation_metrics = previous_validation
                potential_scale = max(
                    arguments.minimum_potential_scale, potential_scale * 0.5
                )
                environment.set_potential_scale(potential_scale)
                _status(
                    f"update={iteration} potential refresh=rejected "
                    f"scale={potential_scale:.4f}"
                )
            potential_optimizer = torch.optim.Adam(
                potential_model.parameters(), lr=arguments.potential_learning_rate
            )
            potential_metrics["refresh_accepted"] = float(accepted)
            potential_metrics["validation_potential_change"] = candidate_change
            potential_metrics.update(
                {
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                }
            )
            test_metrics = evaluate_potential(
                potential_model,
                potential_dataset.test,
                arguments.maximum_steps,
                device,
                history_length=arguments.history_length,
            )
            potential_metrics.update(
                {f"test_{name}": value for name, value in test_metrics.items()}
            )
            snapshot = FrozenPotential(potential_model)
            environment.set_potential(
                make_batched_potential_function(
                    snapshot, arguments.maximum_steps, arguments.history_length
                )
            )
        scores = " ".join(f"{left}:{right}" for left, right in games) or "-"
        print(
            f"update={iteration} games={scores} "
            f"policy={losses['policy']:.4f} value={losses['value']:.4f} "
            f"entropy={losses['entropy']:.4f} "
            f"potential={potential_metrics['outcome']:.4f} "
            f"oda={potential_metrics['oda']:.4f} "
            f"symmetry={potential_metrics['symmetry']:.4f} "
            f"val_acc={potential_metrics.get('validation_outcome_accuracy', 0.0):.4f} "
            f"val_ece={potential_metrics.get('validation_calibration_error', 0.0):.4f} "
            f"beta={potential_scale:.4f}",
            flush=True,
        )
        log_record = {
            "update": iteration,
            "games": games,
            "policy": losses,
            "potential": dict(potential_metrics),
            "diagnostics": diagnostics,
        }

        checkpoint = {
            "model": policy.state_dict(),
            "frame_size": frame_size,
            "action_count": environment.action_count,
            "maximum_steps": arguments.maximum_steps,
            "history_length": arguments.history_length,
            "environments": arguments.environments,
            "potential_ensemble_size": arguments.potential_ensemble_size,
            "potential_batches_per_epoch": arguments.potential_batches_per_epoch,
            "potential_scale": potential_scale,
            "feature_scaling": "fixed",
            "oda_horizon": ODA_HORIZON,
            "oda_stable_steps": ODA_STABLE_STEPS,
            "training_group": arguments.training_group,
            "updates": iteration,
            "potential_model": potential_model.state_dict(),
            "potential_updates": len(potential_dataset),
            "dataset_sizes": potential_dataset.sizes().__dict__,
            "potential_metrics": potential_metrics,
            "diagnostics": diagnostics,
            "dataset_file": output_dataset.name,
            "optimizer": optimizer.state_dict(),
            "potential_optimizer": potential_optimizer.state_dict(),
        }
        _status(f"update={iteration} checkpoint saving")
        _atomic_torch_save(potential_dataset.state_dict(), output_dataset)
        _atomic_torch_save(checkpoint, arguments.checkpoint)
        _append_log(log_path, log_record)
        _status(f"update={iteration} checkpoint saved={arguments.checkpoint}")


if __name__ == "__main__":
    main()
