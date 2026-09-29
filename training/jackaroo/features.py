"""Deterministic conversion of the public model observation to tensors."""

from __future__ import annotations

from typing import Any

import torch


Observation = dict[str, Any]


def _one_hot(index: int, size: int) -> list[float]:
    result = [0.0] * size
    if 0 <= index < size:
        result[index] = 1.0
    return result


def encode(observation: Observation) -> torch.Tensor:
    """Encode the public football state used by the Jackaroo policy.

    Absolute match time and the engine step remain environment controls. No
    training-only player selection or hidden engine state is added here.
    """

    values: list[float] = []
    values.extend(observation["ball_position"])
    values.extend(observation["ball_velocity"])
    values.extend(observation["ball_rotation"])

    for team in observation["teams"]:
        for player in team:
            values.extend(player["position"])
            values.extend(player["velocity"])
            values.extend(player["facing"])
            values.extend(player["formation_position"])
            values.extend(player["dynamic_formation_position"])
            values.append(player["tired_factor"])
            values.append(player["role"] / 10.0)
            values.append(player["dynamic_role"] / 10.0)
            values.append(player["function_type"] / 14.0)
            values.append(player["action_frame"] / 1000.0)
            values.append(player["touch_frame"] / 1000.0)
            values.append(player["possession_duration_ms"] / 10000.0)
            values.append(player["time_to_ball_ms"] / 10000.0)
            values.append(float(player["has_card"]))
            values.append(float(player["is_active"]))
            values.append(float(player["touch_pending"]))

    for team in observation["team_state"]:
        values.extend(
            (
                team["possession_amount"],
                team["fading_possession_amount"],
                team["offside_trap_x"] / 100.0,
                team["time_to_ball_ms"] / 10000.0,
            )
        )
        values.extend(_one_hot(team["designated_possession_player"] + 1, 12))

    values.extend(goal / 5.0 for goal in observation["goals"])
    values.extend(_one_hot(observation["game_mode"], 7))
    values.extend(_one_hot(observation["set_piece_team"] + 1, 3))
    values.extend(_one_hot(observation["set_piece_taker"] + 1, 12))
    values.extend(_one_hot(observation["ball_owned_team"] + 1, 3))
    values.extend(_one_hot(observation["ball_owned_player"] + 1, 12))
    values.extend(_one_hot(observation["last_touch_team"] + 1, 3))
    values.extend(_one_hot(observation["last_touch_player"] + 1, 12))
    values.append(float(observation["is_in_play"]))

    for player in observation["sticky_actions"]:
        values.extend(float(action) for action in player)

    return torch.tensor(values, dtype=torch.float32)
