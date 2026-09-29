"""Clipped Proximal Policy Optimization for vector football environments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn

from .attention import TensorFrame, batch_tensor_histories, tensorize
from .env import VectorFootballEnv
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


def collect_vector_rollout(
    environment: VectorFootballEnv,
    policy: ActorCritic,
    observations: list[dict],
    steps: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    policy_histories: list[list[TensorFrame]] | None = None,
    history_length: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[Rollout, list[dict], list[tuple[int, int]]]:
    """Collect concurrent trajectories and preserve each environment's GAE."""

    if steps <= 0:
        raise ValueError("rollout steps must be positive")
    if len(observations) != environment.count:
        raise ValueError("one observation is required per environment")
    device = next(policy.parameters()).device
    if policy_histories is None:
        policy_histories = [[] for _ in range(environment.count)]
    if len(policy_histories) != environment.count:
        raise ValueError("history count must match environment count")

    encoded_parts = [[] for _ in range(environment.count)]
    own_parts = [[] for _ in range(environment.count)]
    opponent_parts = [[] for _ in range(environment.count)]
    context_parts = [[] for _ in range(environment.count)]
    anchor_parts = [[] for _ in range(environment.count)]
    opponent_anchor_parts = [[] for _ in range(environment.count)]
    action_parts = [[] for _ in range(environment.count)]
    log_probability_parts = [[] for _ in range(environment.count)]
    value_parts = [[] for _ in range(environment.count)]
    reward_parts = [[] for _ in range(environment.count)]
    terminated_parts = [[] for _ in range(environment.count)]
    task_reward_total = 0.0
    potential_reward_total = 0.0
    telescoping_error_total = 0.0
    completed_games: list[tuple[int, int]] = []
    transitions = 0
    report_interval = max(environment.count, min(512, max(steps // 8, 1)))
    next_report = report_interval

    while transitions < steps:
        for index, observation in enumerate(observations):
            policy_histories[index].append(
                tensorize(observation, environment.maximum_steps)
            )
            if len(policy_histories[index]) > history_length:
                del policy_histories[index][:-history_length]
        inputs = batch_tensor_histories(
            policy_histories, history_length, device
        )
        encoded, own, opponent, context, anchors, opponent_anchors = inputs
        with torch.inference_mode():
            distribution, value = policy.distributions(*inputs)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        next_observations, rewards, terminated, infos = environment.step(
            action.tolist()
        )
        for index in range(environment.count):
            reward = rewards[index]
            if not math.isfinite(reward):
                raise FloatingPointError("rollout reward is not finite")
            encoded_parts[index].append(encoded[index])
            own_parts[index].append(own[index])
            opponent_parts[index].append(opponent[index])
            context_parts[index].append(context[index])
            anchor_parts[index].append(anchors[index])
            opponent_anchor_parts[index].append(opponent_anchors[index])
            action_parts[index].append(action[index])
            log_probability_parts[index].append(log_probability[index])
            value_parts[index].append(value[index])
            reward_parts[index].append(reward)
            terminated_parts[index].append(terminated[index])
            task_reward_total += infos[index]["reward_components"]["task"]
            potential_reward_total += infos[index]["reward_components"]["potential"]
            telescoping_error_total += abs(infos[index]["potential_telescoping_error"])
            if terminated[index]:
                completed_games.append(tuple(infos[index]["goals"]))
                policy_histories[index].clear()
                next_observations[index] = environment.reset_one(index)
        observations = next_observations
        transitions += environment.count
        if progress is not None and transitions >= next_report:
            match_steps = [observation["step"] for observation in observations]
            progress(
                f"steps={transitions} target={steps} "
                f"match_step={min(match_steps)}-{max(match_steps)} "
                f"games={len(completed_games)}"
            )
            next_report += report_interval

    with torch.inference_mode():
        next_inputs = batch_tensor_histories(
            [
                policy_histories[index]
                + [tensorize(observations[index], environment.maximum_steps)]
                for index in range(environment.count)
            ],
            history_length,
            device,
        )
        _, next_values = policy.distributions(*next_inputs)

    advantage_parts = []
    flat_values = []
    for environment_index in range(environment.count):
        values = torch.stack(value_parts[environment_index])
        advantages = torch.zeros_like(values)
        advantage = torch.tensor(0.0, device=device)
        next_value = next_values[environment_index]
        for index in reversed(range(len(reward_parts[environment_index]))):
            continues = 0.0 if terminated_parts[environment_index][index] else 1.0
            delta = (
                reward_parts[environment_index][index]
                + gamma * next_value * continues
                - values[index]
            )
            advantage = delta + gamma * gae_lambda * continues * advantage
            advantages[index] = advantage
            next_value = values[index]
        advantage_parts.append(advantages)
        flat_values.append(values)

    def stack(parts: list[list[torch.Tensor]]) -> torch.Tensor:
        return torch.cat([torch.stack(part) for part in parts])

    advantages = torch.cat(advantage_parts)
    values = torch.cat(flat_values)
    encoded = stack(encoded_parts)
    for name, tensor in (
        ("encoded observations", encoded),
        ("values", values),
        ("advantages", advantages),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"rollout {name} contain NaN or Inf")
    return (
        Rollout(
            encoded=encoded,
            own=stack(own_parts),
            opponent=stack(opponent_parts),
            context=stack(context_parts),
            anchor_indices=stack(anchor_parts),
            opponent_anchor_indices=stack(opponent_anchor_parts),
            actions=stack(action_parts),
            old_log_probabilities=stack(log_probability_parts),
            advantages=advantages,
            returns=advantages + values,
            task_reward_total=task_reward_total,
            potential_reward_total=potential_reward_total,
            telescoping_error_total=telescoping_error_total,
        ),
        observations,
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
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    sample_count = rollout.encoded.shape[0]
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    updates = 0
    policy.train()
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(
            sample_count, device=rollout.encoded.device
        ).split(batch_size):
            distribution, value = policy.distributions(
                rollout.encoded[indices],
                rollout.own[indices],
                rollout.opponent[indices],
                rollout.context[indices],
                rollout.anchor_indices[indices],
                rollout.opponent_anchor_indices[indices],
            )
            log_probability = distribution.log_prob(rollout.actions[indices])
            ratio = (log_probability - rollout.old_log_probabilities[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[
                indices
            ]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, rollout.returns[indices])
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
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
