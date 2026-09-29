"""Behavior-cloning initialization from complete built-in AI matches."""

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
DemonstrationMatch = list[Demonstration]


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
    matches: int,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> list[DemonstrationMatch]:
    """Collect every observable built-in action from complete matches."""

    demonstrations: list[DemonstrationMatch] = []
    completed_matches = 0
    while completed_matches < matches:
        first = completed_matches
        seeds = [
            (seed + first + index) & 0xFFFFFFFF
            for index in range(environment.count)
        ]
        sides = [(first + index) % 2 == 0 for index in range(environment.count)]
        if progress is not None:
            progress(
                f"batch={first + 1}-{min(first + environment.count, matches)}/"
                f"{matches} start"
            )
        observations = environment.reset_builtin(seeds, sides)
        match_demonstrations: list[DemonstrationMatch] = [
            [] for _ in range(environment.count)
        ]
        match_done = [False] * environment.count
        next_progress_step = 500
        while not all(match_done):
            previous_observations = observations
            observations, current_match_done = environment.step_builtin()
            for index, observation in enumerate(observations):
                if not match_done[index]:
                    previous = previous_observations[index]
                    if previous["is_in_play"]:
                        match_demonstrations[index].append(
                            (
                                tensorize(previous),
                                project_builtin_action(previous, observation),
                            )
                        )
                match_done[index] = match_done[index] or current_match_done[index]
            minimum_step = min(observation["step"] for observation in observations)
            if progress is not None and minimum_step >= next_progress_step:
                progress(
                    f"batch={first + 1}-{min(first + environment.count, matches)}/"
                    f"{matches} step={minimum_step}/{environment.maximum_steps}"
                )
                next_progress_step += 500
        remaining = matches - completed_matches
        for index, match in enumerate(match_demonstrations[:remaining]):
            demonstrations.append(match)
            completed_matches += 1
            if progress is not None:
                goals = observations[index]["goals"]
                progress(
                    f"match={completed_matches}/{matches} complete "
                    f"score={goals[0]}:{goals[1]} samples={len(match)}"
                )
    return demonstrations


def pretrain_policy(
    policy: ActorCritic,
    demonstrations: Sequence[DemonstrationMatch],
    history_length: int,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Initialize the actor from all actions in complete built-in matches."""

    samples: list[tuple[list[TensorFrame], int]] = []
    for match in demonstrations:
        for index, (_, action) in enumerate(match):
            history = [
                frame
                for frame, _ in match[
                    max(0, index - history_length + 1):index + 1
                ]
            ]
            samples.append((history, action))
    if not samples or epochs <= 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "accuracy": 0.0,
            "samples": float(len(samples)),
            "matches": float(len(demonstrations)),
        }

    encoded_parts = []
    own_parts = []
    opponent_parts = []
    context_parts = []
    anchor_parts = []
    opponent_anchor_parts = []
    sample_count = len(samples)
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    encoding_batch_size = max(batch_size, 512)
    encoding_batch_count = math.ceil(sample_count / encoding_batch_size)
    encoding_report_interval = max(encoding_batch_count // 10, 1)
    for batch_number, start in enumerate(
        range(0, sample_count, encoding_batch_size), start=1
    ):
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
        if progress is not None and (
            batch_number % encoding_report_interval == 0
            or batch_number == encoding_batch_count
        ):
            progress(
                f"cache samples={min(start + encoding_batch_size, sample_count)}/"
                f"{sample_count}"
            )
    cached_inputs_list = []
    for parts in (
        encoded_parts,
        own_parts,
        opponent_parts,
        context_parts,
        anchor_parts,
        opponent_anchor_parts,
    ):
        cached_inputs_list.append(torch.cat(parts, dim=0))
        parts.clear()
    cached_inputs = tuple(cached_inputs_list)
    if progress is not None:
        progress(f"cache complete samples={sample_count}")
    counts = torch.bincount(targets, minlength=policy.action_head.out_features).float()
    class_weights = torch.zeros_like(counts)
    observed = counts > 0
    class_weights[observed] = counts[observed].rsqrt()
    class_weights[observed] /= class_weights[observed].mean()
    class_weights = class_weights.to(device)
    del samples

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    final_loss = 0.0
    final_policy_loss = 0.0
    final_correct = 0
    final_count = 0
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(sample_count)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_correct = 0
        total_count = 0
        policy.train()
        batch_count = math.ceil(sample_count / batch_size)
        report_interval = max(batch_count // 4, 1)
        for batch_number, start in enumerate(
            range(0, sample_count, batch_size), start=1
        ):
            indices = permutation[start:start + batch_size]
            batch_inputs = tuple(tensor[indices].to(device) for tensor in cached_inputs)
            batch_targets = targets[indices].to(device)
            logits = policy.action_logits(*batch_inputs)
            policy_loss = nn.functional.cross_entropy(
                logits, batch_targets, weight=class_weights
            )
            loss = policy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            count = batch_targets.numel()
            total_loss += loss.item() * count
            total_policy_loss += policy_loss.item() * count
            total_correct += (
                logits.argmax(dim=-1) == batch_targets
            ).sum().item()
            total_count += count
            if progress is not None and (
                batch_number % report_interval == 0 or batch_number == batch_count
            ):
                progress(
                    f"epoch={epoch}/{epochs} "
                    f"batch={batch_number}/{batch_count}"
                )
        final_loss = total_loss / total_count
        final_policy_loss = total_policy_loss / total_count
        final_correct = total_correct
        final_count = total_count
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} policy={final_policy_loss:.4f} "
                f"accuracy={total_correct / total_count:.4f}"
            )
    return {
        "loss": final_loss,
        "policy_loss": final_policy_loss,
        "accuracy": final_correct / final_count,
        "samples": float(sample_count),
        "matches": float(len(demonstrations)),
    }
