"""Train a single-agent football policy with PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import FootballEnv
from .features import encode
from .network import ActorCritic
from .ppo import collect_rollout, update


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/ppo.pt")
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    environment = FootballEnv(arguments.maximum_steps)
    observation = environment.reset(arguments.seed)
    observation_size = encode(observation, arguments.maximum_steps).numel()
    policy = ActorCritic(
        observation_size, environment.player_count, environment.action_count
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=arguments.learning_rate)

    for iteration in range(1, arguments.updates + 1):
        rollout, observation, games = collect_rollout(
            environment,
            policy,
            observation,
            arguments.steps_per_update,
        )
        losses = update(policy, optimizer, rollout)
        scores = " ".join(f"{left}:{right}" for left, right in games) or "-"
        print(
            f"update={iteration} games={scores} "
            f"policy={losses['policy']:.4f} value={losses['value']:.4f} "
            f"entropy={losses['entropy']:.4f}",
            flush=True,
        )

        arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": policy.state_dict(),
                "observation_size": observation_size,
                "player_count": environment.player_count,
                "action_count": environment.action_count,
                "maximum_steps": arguments.maximum_steps,
                "updates": iteration,
            },
            arguments.checkpoint,
        )


if __name__ == "__main__":
    main()
