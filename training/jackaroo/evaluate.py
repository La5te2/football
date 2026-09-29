"""Evaluate a saved PPO policy against the built-in team AI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from .env import FootballEnv
from .attention import batch_tensor_histories, tensorize
from .network import ActorCritic, POLICY_ARCHITECTURE, POLICY_OBJECTIVE


def evaluate_policy(
    policy: ActorCritic,
    maximum_steps: int,
    history_length: int,
    games: int,
    seed: int,
    progress: Callable[[int, tuple[int, int]], None] | None = None,
) -> dict[str, float]:
    """Evaluate one policy over complete matches and next-goal episodes."""

    if games < 0:
        raise ValueError("games must be nonnegative")
    if games == 0:
        return {
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "goal_difference": 0.0,
            "segment_wins": 0.0,
            "segment_draws": 0.0,
            "segment_losses": 0.0,
            "mean_segment_steps": 0.0,
        }

    device = next(policy.parameters()).device
    environment = FootballEnv(maximum_steps, seed)
    wins = draws = losses = goal_difference = 0
    segment_wins = segment_draws = segment_losses = segment_steps = 0
    policy.eval()
    for game in range(games):
        observation = environment.reset((seed + game // 2) & 0xFFFFFFFF)
        history = []
        match_done = False
        info = {"goals": (0, 0)}
        while not match_done:
            action = 0
            if observation["is_in_play"]:
                history.append(tensorize(observation, maximum_steps))
                if len(history) > history_length:
                    del history[:-history_length]
                with torch.inference_mode():
                    logits, _ = policy(
                        *batch_tensor_histories([history], history_length, device)
                    )
                action = logits[0].argmax().item()
            observation, _, segment_done, info = environment.step(action)
            if segment_done:
                result = info["segment_result"]
                segment_wins += result == 1
                segment_draws += result == 0
                segment_losses += result == -1
                segment_steps += info["segment_steps"]
                history.clear()
            match_done = info["match_done"]
        goals = tuple(info["goals"])
        if progress is not None:
            progress(game + 1, goals)
        own, opponent = goals
        wins += own > opponent
        draws += own == opponent
        losses += own < opponent
        goal_difference += own - opponent
    segment_count = segment_wins + segment_draws + segment_losses
    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "goal_difference": goal_difference / games,
        "segment_wins": float(segment_wins),
        "segment_draws": float(segment_draws),
        "segment_losses": float(segment_losses),
        "mean_segment_steps": segment_steps / segment_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    history_length = checkpoint.get("history_length", 1)
    metrics = evaluate_policy(
        policy,
        checkpoint["maximum_steps"],
        history_length,
        arguments.games,
        arguments.seed,
        progress=lambda game, goals: print(
            f"game={game} score={goals[0]}:{goals[1]}"
        ),
    )
    print(
        f"summary matches={int(metrics['wins'])}/{int(metrics['draws'])}/"
        f"{int(metrics['losses'])} "
        f"mean_goal_difference={metrics['goal_difference']:.3f} "
        f"segments={int(metrics['segment_wins'])}/"
        f"{int(metrics['segment_draws'])}/{int(metrics['segment_losses'])} "
        f"mean_segment_steps={metrics['mean_segment_steps']:.1f}"
    )


if __name__ == "__main__":
    main()
