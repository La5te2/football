"""Evaluate a saved PPO policy against the built-in team AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import FootballEnv
from .features import encode
from .network import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()

    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    environment = FootballEnv(checkpoint["maximum_steps"])
    policy = ActorCritic(
        checkpoint["observation_size"], checkpoint["action_count"]
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    for game in range(arguments.games):
        observation = environment.reset(arguments.seed + game // 2)
        terminated = False
        info = {"goals": (0, 0)}
        while not terminated:
            state = encode(observation, environment.maximum_steps).to(device)
            with torch.no_grad():
                action_logits, _ = policy(state)
                action = action_logits.argmax().item()
            observation, _, terminated, info = environment.step(action)
        print(f"game={game + 1} score={info['goals'][0]}:{info['goals'][1]}")


if __name__ == "__main__":
    main()
