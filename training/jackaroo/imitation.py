"""Behavior-cloning bootstrap from built-in AI match trajectories."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import nn

from .attention import TensorFrame, batch_tensor_histories, tensorize
from .env import Observation, VectorFootballEnv
from .network import ActorCritic


FUNCTION_SHORT_PASS = 4
FUNCTION_LONG_PASS = 5
FUNCTION_HIGH_PASS = 6
FUNCTION_SHOT = 8
FUNCTION_INTERFERE = 11
FUNCTION_SLIDING = 13

ACTION_IDLE = 0
ACTION_LONG_PASS = 9
ACTION_HIGH_PASS = 10
ACTION_SHORT_PASS = 11
ACTION_SHOT = 12
ACTION_SLIDING = 14
ACTION_PRESSURE = 15
Demonstration = tuple[TensorFrame, int]


def _direction_action(x: float, y: float) -> int:
    """Quantize a canonical movement vector into the engine's eight directions."""

    if math.hypot(x, y) < 0.01:
        return ACTION_IDLE
    angle = math.atan2(y, x)
    octant = int(round(angle / (math.pi / 4.0))) % 8
    # Engine order starts at left, then rotates clockwise through top-left.
    return (5, 4, 3, 2, 1, 8, 7, 6)[octant]


def project_builtin_action(previous: Observation, current: Observation) -> int:
    """Project the built-in policy's visible result onto the 32-action interface.

    Eliza produces continuous commands rather than interface button indices.
    The projection uses the next state's executed animation and movement for the
    player designated in the source observation.
    """

    player = int(previous["team_state"][0]["designated_possession_player"])
    if not 0 <= player < 11:
        return ACTION_IDLE
    previous_state = previous["teams"][0][player]
    state = current["teams"][0][player]
    function = int(state["function_type"])
    action_started = (
        function != int(previous_state["function_type"])
        or int(state["action_frame"]) < int(previous_state["action_frame"])
    )
    if action_started:
        if function == FUNCTION_SHORT_PASS:
            return ACTION_SHORT_PASS
        if function == FUNCTION_LONG_PASS:
            return ACTION_LONG_PASS
        if function == FUNCTION_HIGH_PASS:
            return ACTION_HIGH_PASS
        if function == FUNCTION_SHOT:
            return ACTION_SHOT
        if function == FUNCTION_SLIDING:
            return ACTION_SLIDING
        if function == FUNCTION_INTERFERE:
            return ACTION_PRESSURE
    velocity = state["velocity"]
    return _direction_action(float(velocity[0]), float(velocity[1]))


def collect_demonstrations(
    environment: VectorFootballEnv,
    games: int,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> list[list[Demonstration]]:
    """Collect encoded built-in actions without retaining observation dictionaries."""

    episodes: list[list[Demonstration]] = []
    while len(episodes) < games:
        first = len(episodes)
        seeds = [
            (seed + first + index) & 0xFFFFFFFF
            for index in range(environment.count)
        ]
        sides = [(first + index) % 2 == 0 for index in range(environment.count)]
        if progress is not None:
            progress(
                f"batch={first + 1}-{min(first + environment.count, games)}/"
                f"{games} start"
            )
        observations = environment.reset_builtin(seeds, sides)
        demonstrations: list[list[Demonstration]] = [
            [] for _ in range(environment.count)
        ]
        terminated = [False] * environment.count
        next_progress_step = 500
        while not all(terminated):
            previous_observations = observations
            observations, current_terminated = environment.step_builtin()
            for index, observation in enumerate(observations):
                if not terminated[index]:
                    demonstrations[index].append(
                        (
                            tensorize(
                                previous_observations[index],
                                environment.maximum_steps,
                            ),
                            project_builtin_action(
                                previous_observations[index], observation
                            ),
                        )
                    )
                terminated[index] = terminated[index] or current_terminated[index]
            minimum_step = min(observation["step"] for observation in observations)
            if progress is not None and minimum_step >= next_progress_step:
                progress(
                    f"batch={first + 1}-{min(first + environment.count, games)}/"
                    f"{games} step={minimum_step}/{environment.maximum_steps}"
                )
                next_progress_step += 500
        remaining = games - len(episodes)
        for index, demonstration in enumerate(demonstrations[:remaining]):
            episodes.append(demonstration)
            if progress is not None:
                goals = observations[index]["goals"]
                progress(
                    f"game={len(episodes)}/{games} complete "
                    f"score={goals[0]}:{goals[1]}"
                )
    return episodes


def pretrain_policy(
    policy: ActorCritic,
    episodes: Sequence[Sequence[Demonstration]],
    history_length: int,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Fit the actor to projected built-in actions before PPO optimization."""

    samples: list[tuple[list[TensorFrame], int]] = []
    for episode in episodes:
        for index, (_, action) in enumerate(episode):
            history = [
                frame
                for frame, _ in episode[
                    max(0, index - history_length + 1):index + 1
                ]
            ]
            samples.append((history, action))
    if not samples or epochs <= 0:
        return {"loss": 0.0, "accuracy": 0.0, "samples": float(len(samples))}

    encoded_parts = []
    own_parts = []
    opponent_parts = []
    context_parts = []
    anchor_parts = []
    opponent_anchor_parts = []
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    encoding_batch_size = max(batch_size, 512)
    for start in range(0, len(samples), encoding_batch_size):
        inputs = batch_tensor_histories(
            [sample[0] for sample in samples[start:start + encoding_batch_size]],
            history_length,
            torch.device("cpu"),
        )
        for destination, tensor in zip(
            (
                encoded_parts,
                own_parts,
                opponent_parts,
                context_parts,
                anchor_parts,
                opponent_anchor_parts,
            ),
            inputs,
        ):
            destination.append(tensor)
    cached_inputs = tuple(
        torch.cat(parts, dim=0)
        for parts in (
            encoded_parts,
            own_parts,
            opponent_parts,
            context_parts,
            anchor_parts,
            opponent_anchor_parts,
        )
    )
    counts = torch.bincount(targets, minlength=policy.action_head.out_features).float()
    class_weights = torch.zeros_like(counts)
    observed = counts > 0
    class_weights[observed] = counts[observed].rsqrt()
    class_weights[observed] /= class_weights[observed].mean()
    class_weights = class_weights.to(device)
    sample_count = len(samples)
    del samples

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    final_loss = 0.0
    final_correct = 0
    final_count = 0
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(sample_count)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        policy.train()
        for start in range(0, sample_count, batch_size):
            indices = permutation[start:start + batch_size]
            batch_inputs = tuple(tensor[indices].to(device) for tensor in cached_inputs)
            batch_targets = targets[indices].to(device)
            logits, _ = policy(*batch_inputs)
            loss = nn.functional.cross_entropy(
                logits, batch_targets, weight=class_weights
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            count = batch_targets.numel()
            total_loss += loss.item() * count
            total_correct += (
                logits.argmax(dim=-1) == batch_targets
            ).sum().item()
            total_count += count
        final_loss = total_loss / total_count
        final_correct = total_correct
        final_count = total_count
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} loss={final_loss:.4f} "
                f"accuracy={total_correct / total_count:.4f}"
            )
    return {
        "loss": final_loss,
        "accuracy": final_correct / final_count,
        "samples": float(sample_count),
    }
