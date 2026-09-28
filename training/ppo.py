"""Minimal clipped Proximal Policy Optimization implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .env import FootballEnv
from .features import encode
from .network import ActorCritic


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def collect_rollout(
    environment: FootballEnv,
    policy: ActorCritic,
    observation: dict,
    steps: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Rollout, dict, list[tuple[int, int]]]:
    device = next(policy.parameters()).device
    observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    terminated_flags = []
    completed_games: list[tuple[int, int]] = []

    for _ in range(steps):
        encoded = encode(observation, environment.maximum_steps).to(device)
        with torch.no_grad():
            action_distribution, value = policy.distributions(encoded)
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action)

        next_observation, reward, terminated, info = environment.step(action.item())
        observations.append(encoded)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward)
        terminated_flags.append(terminated)

        observation = next_observation
        if terminated:
            completed_games.append(tuple(info["goals"]))
            observation = environment.reset()

    with torch.no_grad():
        if terminated_flags[-1]:
            next_value = torch.tensor(0.0)
        else:
            next_encoded = encode(observation, environment.maximum_steps).to(device)
            _, next_value = policy.distributions(next_encoded)

    advantages = torch.zeros(steps, device=device)
    advantage = torch.tensor(0.0, device=device)
    for index in reversed(range(steps)):
        continues = 0.0 if terminated_flags[index] else 1.0
        delta = (
            rewards[index]
            + gamma * next_value * continues
            - values[index]
        )
        advantage = delta + gamma * gae_lambda * continues * advantage
        advantages[index] = advantage
        next_value = values[index]

    value_tensor = torch.stack(values)
    return (
        Rollout(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            old_log_probabilities=torch.stack(log_probabilities),
            advantages=advantages,
            returns=advantages + value_tensor,
        ),
        observation,
        completed_games,
    )


def update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 4,
    batch_size: int = 256,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> dict[str, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    sample_count = rollout.observations.shape[0]
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        for indices in torch.randperm(
            sample_count, device=rollout.observations.device
        ).split(batch_size):
            action_distribution, value = policy.distributions(
                rollout.observations[indices]
            )
            log_probability = action_distribution.log_prob(rollout.actions[indices])
            ratio = (log_probability - rollout.old_log_probabilities[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[
                indices
            ]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, rollout.returns[indices])
            entropy = action_distribution.entropy().mean()
            loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            updates += 1

    return {name: value / updates for name, value in totals.items()}
