"""Deterministic conversion from native match dictionaries to policy tensors."""

from __future__ import annotations

from typing import Any

import torch


def _one_hot(index: int, size: int) -> list[float]:
    result = [0.0] * size
    if 0 <= index < size:
        result[index] = 1.0
    return result


def encode(observation: dict[str, Any], maximum_steps: int) -> torch.Tensor:
    values: list[float] = []
    values.extend(observation["ball_position"])
    values.extend(observation["ball_velocity"])
    values.extend(observation["ball_rotation"])
    for team in observation["teams"]:
        for player in team:
            values.extend(player["position"])
            values.extend(player["velocity"])
            values.append(player["tired_factor"])
            values.append(player["role"] / 9.0)
            values.append(float(player["has_card"]))
            values.append(float(player["is_active"]))
    values.extend(goal / 5.0 for goal in observation["goals"])
    values.extend(_one_hot(observation["game_mode"], 7))
    values.extend(_one_hot(observation["ball_owned_team"] + 1, 3))
    values.extend(_one_hot(observation["ball_owned_player"] + 1, 12))
    values.append(observation["step"] / maximum_steps)
    values.extend(float(value) for value in observation["sticky_actions"])
    return torch.tensor(values, dtype=torch.float32)


def active_player_mask(observation: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [player["is_active"] for player in observation["teams"][0]],
        dtype=torch.bool,
    )
