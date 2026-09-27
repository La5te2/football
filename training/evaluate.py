"""Evaluate a saved PPO policy against the built-in team AI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import FootballEnv
from .features import active_player_mask, encode
from .network import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    environment = FootballEnv(checkpoint["maximum_steps"])
    policy = ActorCritic(
        checkpoint["observation_size"],
        checkpoint["player_count"],
        checkpoint["action_count"],
    )
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    for game in range(arguments.games):
        observation = environment.reset(arguments.seed + game)
        terminated = False
        info = {"goals": (0, 0)}
        while not terminated:
            state = encode(observation, environment.maximum_steps)
            mask = active_player_mask(observation)
            with torch.no_grad():
                player_logits, action_logits, _ = policy(state)
                player_logits = player_logits.masked_fill(~mask, -torch.inf)
                player = player_logits.argmax().item()
                action = action_logits.argmax().item()
            observation, _, terminated, info = environment.step(player, action)
        print(f"game={game + 1} score={info['goals'][0]}:{info['goals'][1]}")


if __name__ == "__main__":
    main()
