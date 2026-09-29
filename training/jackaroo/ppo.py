"""Minimal clipped Proximal Policy Optimization implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn

from .env import FootballEnv
from .attention import batch_histories
from .dataset import Episode
from .network import ActorCritic


@dataclass
class Rollout:
    encoded: torch.Tensor
    own: torch.Tensor
    opponent: torch.Tensor
    context: torch.Tensor
    anchor_indices: torch.Tensor
    opponent_anchor_indices: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    task_reward_total: float
    potential_reward_total: float
    telescoping_error_total: float


def collect_rollout(
    environment: FootballEnv,
    policy: ActorCritic,
    observation: dict,
    steps: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    episode_history: list[dict] | None = None,
    policy_history: list[dict] | None = None,
    history_length: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[Rollout, dict, list[tuple[int, int]], list[Episode]]:
    if steps <= 0:
        raise ValueError("rollout steps must be positive")
    device = next(policy.parameters()).device
    encoded_observations = []
    own_observations = []
    opponent_observations = []
    context_observations = []
    anchor_observations = []
    opponent_anchor_observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    terminated_flags = []
    task_reward_total = 0.0
    potential_reward_total = 0.0
    telescoping_error_total = 0.0
    completed_games: list[tuple[int, int]] = []
    completed_episodes: list[Episode] = []
    if episode_history is None:
        episode_history = []
    if policy_history is None:
        policy_history = []
    report_interval = max(1, min(500, steps // 8))

    while len(rewards) < steps or episode_history:
        episode_history.append(observation)
        policy_history.append(observation)
        (
            encoded,
            own,
            opponent,
            context,
            anchors,
            opponent_anchors,
        ) = batch_histories(
            [policy_history], environment.maximum_steps, history_length, device
        )
        with torch.no_grad():
            action_distribution, value = policy.distributions(
                encoded, own, opponent, context, anchors, opponent_anchors
            )
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action)

        next_observation, reward, terminated, info = environment.step(action.item())
        if not math.isfinite(reward):
            raise FloatingPointError("rollout reward is not finite")
        encoded_observations.append(encoded.squeeze(0))
        own_observations.append(own.squeeze(0))
        opponent_observations.append(opponent.squeeze(0))
        context_observations.append(context.squeeze(0))
        anchor_observations.append(anchors.squeeze(0))
        opponent_anchor_observations.append(opponent_anchors.squeeze(0))
        actions.append(action.squeeze(0))
        log_probabilities.append(log_probability.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(reward)
        task_reward_total += info["reward_components"]["score"]
        potential_reward_total += info["reward_components"]["potential"]
        telescoping_error_total += abs(info["potential_telescoping_error"])
        terminated_flags.append(terminated)
        if progress is not None and (
            len(rewards) % report_interval == 0 or terminated
        ):
            phase = "collecting" if len(rewards) < steps else "finishing-match"
            progress(
                f"phase={phase} steps={len(rewards)} target={steps} "
                f"match_step={next_observation['step']}"
            )

        observation = next_observation
        if terminated:
            completed_games.append(tuple(info["goals"]))
            episode_history.append(next_observation)
            own_goals, opponent_goals = info["goals"]
            outcome = 0 if own_goals > opponent_goals else 1 if own_goals == opponent_goals else 2
            completed_episodes.append(
                Episode(observations=list(episode_history), outcome=outcome)
            )
            episode_history.clear()
            policy_history.clear()
            observation = environment.reset()

    with torch.no_grad():
        if terminated_flags[-1]:
            next_value = torch.tensor(0.0, device=device)
        else:
            next_inputs = batch_histories(
                [policy_history + [observation]],
                environment.maximum_steps,
                history_length,
                device,
            )
            _, next_value = policy.distributions(*next_inputs)
            next_value = next_value.squeeze(0)

    actual_steps = len(rewards)
    advantages = torch.zeros(actual_steps, device=device)
    advantage = torch.tensor(0.0, device=device)
    for index in reversed(range(actual_steps)):
        continues = 0.0 if terminated_flags[index] else 1.0
        delta = rewards[index] + gamma * next_value * continues - values[index]
        advantage = delta + gamma * gae_lambda * continues * advantage
        advantages[index] = advantage
        next_value = values[index]

    value_tensor = torch.stack(values)
    for name, tensor in (
        ("encoded observations", torch.stack(encoded_observations)),
        ("values", value_tensor),
        ("advantages", advantages),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"rollout {name} contain NaN or Inf")
    return (
        Rollout(
            encoded=torch.stack(encoded_observations),
            own=torch.stack(own_observations),
            opponent=torch.stack(opponent_observations),
            context=torch.stack(context_observations),
            anchor_indices=torch.stack(anchor_observations),
            opponent_anchor_indices=torch.stack(opponent_anchor_observations),
            actions=torch.stack(actions),
            old_log_probabilities=torch.stack(log_probabilities),
            advantages=advantages,
            returns=advantages + value_tensor,
            task_reward_total=task_reward_total,
            potential_reward_total=potential_reward_total,
            telescoping_error_total=telescoping_error_total,
        ),
        observation,
        completed_games,
        completed_episodes,
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
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    sample_count = rollout.encoded.shape[0]
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    updates = 0

    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(
            sample_count, device=rollout.encoded.device
        ).split(batch_size):
            action_distribution, value = policy.distributions(
                rollout.encoded[indices],
                rollout.own[indices],
                rollout.opponent[indices],
                rollout.context[indices],
                rollout.anchor_indices[indices],
                rollout.opponent_anchor_indices[indices],
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
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is NaN or Inf")

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is NaN or Inf")
            optimizer.step()

            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            updates += 1
        if progress is not None:
            progress(f"epoch={epoch}/{epochs}")

    return {name: value / updates for name, value in totals.items()}
