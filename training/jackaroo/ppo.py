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
    reward_total: float


def collect_vector_rollout(
    environment: VectorFootballEnv,
    policy: ActorCritic,
    observations: list[dict],
    steps: int,
    policy_histories: list[list[TensorFrame]] | None = None,
    history_length: int = 4,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    Rollout,
    list[dict],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Collect complete next-goal episodes under one frozen policy."""

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
    segment_done_parts = [[] for _ in range(environment.count)]
    reward_total = 0.0
    completed_segments: list[tuple[int, int]] = []
    completed_matches: list[tuple[int, int]] = []
    retained_transitions = 0
    simulated_transitions = 0
    wave = 0
    report_interval = max(environment.count, min(512, max(steps // 8, 1)))
    next_report = report_interval
    trajectory_parts = (
        encoded_parts,
        own_parts,
        opponent_parts,
        context_parts,
        anchor_parts,
        opponent_anchor_parts,
        action_parts,
        log_probability_parts,
        value_parts,
        reward_parts,
        segment_done_parts,
    )

    while retained_transitions < steps:
        wave += 1
        active = [True] * environment.count
        segment_starts = [len(parts) for parts in reward_parts]
        while any(active):
            active_indices = [
                index for index, enabled in enumerate(active) if enabled
            ]
            inference_histories = []
            for index in active_indices:
                observation = observations[index]
                frame = tensorize(observation)
                if observation["is_in_play"]:
                    policy_histories[index].append(frame)
                    if len(policy_histories[index]) > history_length:
                        del policy_histories[index][:-history_length]
                    inference_histories.append(policy_histories[index])
                else:
                    inference_histories.append(policy_histories[index] + [frame])
            inputs = batch_tensor_histories(
                inference_histories, history_length, device
            )
            encoded, own, opponent, context, anchors, opponent_anchors = inputs
            with torch.inference_mode():
                distribution, value = policy.distributions(*inputs)
                action = distribution.sample()
                log_probability = distribution.log_prob(action)

            actions = [0] * environment.count
            for local_index, environment_index in enumerate(active_indices):
                actions[environment_index] = action[local_index].item()
            next_observations, rewards, segment_done, infos = environment.step(
                actions, active
            )
            for local_index, index in enumerate(active_indices):
                reward = rewards[index]
                if not math.isfinite(reward):
                    raise FloatingPointError("rollout reward is not finite")
                if infos[index]["action_applied"]:
                    encoded_parts[index].append(encoded[local_index])
                    own_parts[index].append(own[local_index])
                    opponent_parts[index].append(opponent[local_index])
                    context_parts[index].append(context[local_index])
                    anchor_parts[index].append(anchors[local_index])
                    opponent_anchor_parts[index].append(
                        opponent_anchors[local_index]
                    )
                    action_parts[index].append(action[local_index])
                    log_probability_parts[index].append(
                        log_probability[local_index]
                    )
                    value_parts[index].append(value[local_index])
                    reward_parts[index].append(reward)
                    segment_done_parts[index].append(segment_done[index])
                    retained_transitions += 1
                    simulated_transitions += 1
                if segment_done[index]:
                    result = infos[index]["segment_result"]
                    if result not in (-1, 0, 1):
                        raise RuntimeError("completed segment has no WDL result")
                    if not infos[index]["action_applied"]:
                        if len(segment_done_parts[index]) == segment_starts[index]:
                            raise RuntimeError(
                                "a completed segment has no trainable transition"
                            )
                        reward_parts[index][-1] += reward
                        segment_done_parts[index][-1] = True
                    completed_segments.append(
                        (int(result), infos[index]["segment_steps"])
                    )
                    if result == 0:
                        segment_length = (
                            len(reward_parts[index]) - segment_starts[index]
                        )
                        for parts in trajectory_parts:
                            del parts[index][segment_starts[index]:]
                        retained_transitions -= segment_length
                    else:
                        reward_total += float(result)
                    active[index] = False
                    policy_histories[index].clear()
                    if infos[index]["match_done"]:
                        completed_matches.append(tuple(infos[index]["goals"]))
                        next_observations[index] = environment.reset_one(index)
            observations = next_observations
            if progress is not None and simulated_transitions >= next_report:
                progress(
                    f"steps={retained_transitions} target={steps} "
                    f"simulated={simulated_transitions} wave={wave} "
                    f"active={sum(active)}/{environment.count} "
                    f"segments={len(completed_segments)} "
                    f"matches={len(completed_matches)}"
                )
                next_report += report_interval
        if progress is not None:
            progress(
                f"wave={wave} complete steps={retained_transitions} "
                f"target={steps} simulated={simulated_transitions} "
                f"segments={len(completed_segments)} "
                f"matches={len(completed_matches)}"
            )

    advantage_parts = []
    return_parts = []
    flat_values = []
    for environment_index in range(environment.count):
        if not value_parts[environment_index]:
            continue
        values = torch.stack(value_parts[environment_index])
        if not segment_done_parts[environment_index][-1]:
            raise RuntimeError("rollout ended with an incomplete segment")
        returns = torch.zeros_like(values)
        outcome = torch.tensor(0.0, device=device)
        for index in reversed(range(len(reward_parts[environment_index]))):
            if segment_done_parts[environment_index][index]:
                outcome = torch.as_tensor(
                    reward_parts[environment_index][index], device=device
                )
            returns[index] = outcome
        return_parts.append(returns)
        advantage_parts.append(returns - values)
        flat_values.append(values)

    def stack(parts: list[list[torch.Tensor]]) -> torch.Tensor:
        return torch.cat([torch.stack(part) for part in parts if part])

    advantages = torch.cat(advantage_parts)
    returns = torch.cat(return_parts)
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
            returns=returns,
            reward_total=reward_total,
        ),
        observations,
        completed_segments,
        completed_matches,
    )


def update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 4,
    batch_size: int = 256,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.001,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    sample_count = rollout.encoded.shape[0]
    totals = {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
    }
    total_samples = 0
    batch_count = math.ceil(sample_count / batch_size)
    report_interval = max(batch_count // 4, 1)
    policy.train()
    for epoch in range(1, epochs + 1):
        epoch_totals = {name: 0.0 for name in totals}
        epoch_samples = 0
        batches = torch.randperm(
            sample_count, device=rollout.encoded.device
        ).split(batch_size)
        for batch_number, indices in enumerate(batches, start=1):
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
            approximate_kl = (
                rollout.old_log_probabilities[indices] - log_probability
            ).mean()
            clip_fraction = (
                (ratio - 1.0).abs() > clip_ratio
            ).float().mean()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[
                indices
            ]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, rollout.returns[indices])
            entropy = distribution.entropy().mean()
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
            count = indices.numel()
            metrics = {
                "policy": policy_loss.item(),
                "value": value_loss.item(),
                "entropy": entropy.item(),
                "kl": approximate_kl.item(),
                "clip_fraction": clip_fraction.item(),
            }
            for name, metric in metrics.items():
                weighted = metric * count
                totals[name] += weighted
                epoch_totals[name] += weighted
            total_samples += count
            epoch_samples += count
            if progress is not None and (
                batch_number % report_interval == 0 or batch_number == batch_count
            ):
                progress(
                    f"epoch={epoch}/{epochs} "
                    f"batch={batch_number}/{batch_count}"
                )
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} complete "
                f"policy={epoch_totals['policy'] / epoch_samples:.4f} "
                f"value={epoch_totals['value'] / epoch_samples:.4f} "
                f"entropy={epoch_totals['entropy'] / epoch_samples:.4f} "
                f"kl={epoch_totals['kl'] / epoch_samples:.5f} "
                f"clipped={epoch_totals['clip_fraction'] / epoch_samples:.4f}"
            )
    return {name: value / total_samples for name, value in totals.items()}
