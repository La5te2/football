"""Evaluate a saved PPO policy against the built-in team AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import FootballEnv
from .attention import batch_tensor_histories, tensorize
from .network import ActorCritic


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
    environment = FootballEnv(checkpoint["maximum_steps"])
    policy = ActorCritic(checkpoint["frame_size"], checkpoint["action_count"]).to(
        device
    )
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    history_length = checkpoint.get("history_length", 1)
    wins = 0
    draws = 0
    losses = 0
    goal_difference = 0

    for game in range(arguments.games):
        observation = environment.reset(arguments.seed + game // 2)
        history = []
        terminated = False
        info = {"goals": (0, 0)}
        while not terminated:
            history.append(tensorize(observation, environment.maximum_steps))
            if len(history) > history_length:
                del history[:-history_length]
            inputs = batch_tensor_histories(
                [history], history_length, device
            )
            with torch.inference_mode():
                action_logits, _ = policy(*inputs)
                action = action_logits[0].argmax().item()
            observation, _, terminated, info = environment.step(action)
        print(f"game={game + 1} score={info['goals'][0]}:{info['goals'][1]}")
        own, opponent = info["goals"]
        wins += own > opponent
        draws += own == opponent
        losses += own < opponent
        goal_difference += own - opponent
    print(
        f"summary wins={wins} draws={draws} losses={losses} "
        f"mean_goal_difference={goal_difference / arguments.games:.3f}"
    )


if __name__ == "__main__":
    main()
