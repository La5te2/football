"""Evaluate a saved PPO policy against the built-in team AI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch

from .env import FootballEnv
from .attention import batch_tensor_frames, tensorize
from .network import ActorCritic, POLICY_ARCHITECTURE, POLICY_OBJECTIVE


def evaluate_policy(
    policy: ActorCritic,
    maximum_steps: int,
    games: int,
    seed: int,
    progress: Callable[[str], None] | None = None,
    record_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate complete matches against built-in AI from paired seeds."""

    if games < 0:
        raise ValueError("games must be nonnegative")
    if games == 0:
        return {
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "goal_difference": 0.0,
            "scores": [],
        }

    device = next(policy.parameters()).device
    environment = FootballEnv(maximum_steps, seed)
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        environment.start_recording(record_path)
    wins = draws = losses = goal_difference = 0
    scores: list[tuple[int, int]] = []
    policy.eval()
    for game in range(games):
        observation = environment.reset(
            (seed + game // 2) & 0xFFFFFFFF, left_team=game % 2 == 0
        )
        state = policy.initial_state(1, device)
        match_done = False
        info = {"goals": (0, 0)}
        next_progress_step = 500
        while not match_done:
            action = 0
            if observation["is_in_play"]:
                with torch.inference_mode():
                    _, output = policy.distributions(
                        *batch_tensor_frames([tensorize(observation)], device),
                        state,
                    )
                    state = output.state
                action = output.logits[0].argmax().item()
            observation, _, segment_done, info = environment.step(action)
            if segment_done:
                state = policy.initial_state(1, device)
            match_done = info["match_done"]
            if progress is not None and observation["step"] >= next_progress_step:
                goals = tuple(info["goals"])
                progress(
                    f"game={game + 1}/{games} "
                    f"step={observation['step']}/{maximum_steps} "
                    f"score={goals[0]}:{goals[1]}"
                )
                next_progress_step += 500
        goals = tuple(info["goals"])
        scores.append(goals)
        if progress is not None:
            progress(
                f"game={game + 1}/{games} complete "
                f"score={goals[0]}:{goals[1]}"
            )
        own, opponent = goals
        wins += own > opponent
        draws += own == opponent
        losses += own < opponent
        goal_difference += own - opponent
    if record_path is not None:
        environment.finish_recording()
    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "goal_difference": goal_difference / games,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--record", type=Path)
    arguments = parser.parse_args()
    if arguments.games <= 0:
        raise ValueError("games must be positive")

    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != POLICY_ARCHITECTURE:
        raise ValueError(
            f"checkpoint architecture={checkpoint.get('architecture')!r} does not "
            f"match {POLICY_ARCHITECTURE!r}"
        )
    if checkpoint.get("objective") != POLICY_OBJECTIVE:
        raise ValueError(
            f"checkpoint objective={checkpoint.get('objective')!r} does not match "
            f"{POLICY_OBJECTIVE!r}"
        )
    policy = ActorCritic(checkpoint["frame_size"], checkpoint["action_count"]).to(
        device
    )
    policy.load_state_dict(checkpoint["model"])
    metrics = evaluate_policy(
        policy,
        checkpoint["maximum_steps"],
        arguments.games,
        arguments.seed,
        progress=print,
        record_path=arguments.record,
    )
    print(
        f"summary matches={int(metrics['wins'])}/{int(metrics['draws'])}/"
        f"{int(metrics['losses'])} "
        f"mean_goal_difference={metrics['goal_difference']:.3f} "
        f"scores={' '.join(f'{left}:{right}' for left, right in metrics['scores'])}"
    )


if __name__ == "__main__":
    main()
