"""Python environment wrapper and task reward for single-agent training."""

from __future__ import annotations

import os
import math
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
if os.name == "nt" and BIN.is_dir():
    os.add_dll_directory(str(BIN))

from . import _gfootball_env as native


Observation = dict[str, Any]

POTENTIAL_DISCOUNT = 0.99
POTENTIAL_SCALE = 0.02
PASS_REWARD = 0.003
FAILED_PASS_PENALTY = 0.003
DANGEROUS_FAILED_PASS_PENALTY = 0.004
ABANDONED_PASS_PENALTY = 0.001
COUNTERPRESS_REWARD = 0.005
COUNTERPRESS_STEPS = 10
PASS_COMPLETION_STEPS = 20


class FootballEnv:
    """One external agent playing either physical side against built-in AI."""

    action_count = native.ENGINE_ACTION_COUNT

    def __init__(self, maximum_steps: int = 3000, seed: int | None = None) -> None:
        data = BIN / "data" if (BIN / "data").is_dir() else ROOT / "engine" / "data"
        fonts = BIN / "fonts" if (BIN / "fonts").is_dir() else ROOT / "engine" / "fonts"
        font = fonts / "AlegreyaSansSC-ExtraBold.ttf"
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._next_left_team = True
        self._native = native.create(str(data), str(font), maximum_steps)
        self._observation: Observation | None = None
        self._stable_possession_team = -1
        self._pending_passer = -1
        self._pending_pass_deadline = -1
        self._pending_pass_danger = 0.0
        self._counterpress_deadline = -1

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._seed_generator.getrandbits(32)
        left_team = self._next_left_team
        self._next_left_team = not self._next_left_team
        self._observation = native.reset(
            self._native, seed & 0xFFFFFFFF, left_team
        )
        self._stable_possession_team = self._observation["ball_owned_team"]
        self._pending_passer = -1
        self._pending_pass_deadline = -1
        self._pending_pass_danger = 0.0
        self._counterpress_deadline = -1
        return self._observation

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count - 1}]")
        previous = self._observation
        decision = [native.ENGINE_ACTION_COUNT] * 11
        player = previous["team_state"][0]["designated_possession_player"]
        if 0 <= player < 11 and previous["teams"][0][player]["is_active"]:
            decision[player] = action
            if (
                action in (9, 10, 11)
                and previous["ball_owned_team"] == 0
                and previous["ball_owned_player"] == player
            ):
                self._pending_passer = player
                self._pending_pass_deadline = previous["step"] + PASS_COMPLETION_STEPS
                self._pending_pass_danger = _clamp(
                    (1.0 - previous["ball_position"][0]) * 0.5
                )
        current, terminated = native.step(self._native, decision)
        reward, reward_components = self._reward(previous, current)
        self._observation = current
        info = {
            "goals": current["goals"],
            "step": current["step"],
            "reward_components": reward_components,
        }
        return current, reward, terminated, info

    def _reward(
        self, previous: Observation, current: Observation
    ) -> tuple[float, dict[str, float]]:
        own_goal = current["goals"][0] - previous["goals"][0]
        opponent_goal = current["goals"][1] - previous["goals"][1]
        score = float(own_goal - opponent_goal)
        if score != 0.0:
            self._stable_possession_team = current["ball_owned_team"]
            self._pending_passer = -1
            self._pending_pass_deadline = -1
            self._pending_pass_danger = 0.0
            self._counterpress_deadline = -1
            return score, {
                "score": score,
                "potential": 0.0,
                "pass": 0.0,
                "transition": 0.0,
            }

        potential = POTENTIAL_SCALE * (
            POTENTIAL_DISCOUNT * _potential(current) - _potential(previous)
        )
        pass_reward = self._pass_reward(current)
        transition_reward = self._transition_reward(current)
        components = {
            "score": 0.0,
            "potential": potential,
            "pass": pass_reward,
            "transition": transition_reward,
        }
        return sum(components.values()), components

    def _pass_reward(self, current: Observation) -> float:
        if self._pending_passer < 0:
            return 0.0
        if current["step"] > self._pending_pass_deadline:
            return self._finish_pass(-ABANDONED_PASS_PENALTY)
        if current["last_touch_team"] == 1 or current["ball_owned_team"] == 1:
            penalty = FAILED_PASS_PENALTY + (
                DANGEROUS_FAILED_PASS_PENALTY * self._pending_pass_danger
            )
            return self._finish_pass(-penalty)
        receiver = current["ball_owned_player"]
        if current["ball_owned_team"] == 0 and receiver != self._pending_passer:
            return self._finish_pass(PASS_REWARD)
        return 0.0

    def _finish_pass(self, reward: float) -> float:
        self._pending_passer = -1
        self._pending_pass_deadline = -1
        self._pending_pass_danger = 0.0
        return reward

    def _transition_reward(self, current: Observation) -> float:
        owner = current["ball_owned_team"]
        if owner not in (0, 1) or owner == self._stable_possession_team:
            if current["step"] > self._counterpress_deadline:
                self._counterpress_deadline = -1
            return 0.0

        reward = 0.0
        if self._stable_possession_team == 0 and owner == 1:
            danger = _clamp((1.0 - current["ball_position"][0]) * 0.5)
            reward -= 0.003 + 0.007 * danger
            self._counterpress_deadline = current["step"] + COUNTERPRESS_STEPS
        elif self._stable_possession_team == 1 and owner == 0:
            if current["step"] <= self._counterpress_deadline:
                reward += COUNTERPRESS_REWARD
            self._counterpress_deadline = -1
        self._stable_possession_team = owner
        return reward


def _potential(observation: Observation) -> float:
    possession = observation["ball_owned_team"]
    possession_score = 1.0 if possession == 0 else 0.0 if possession == 1 else 0.5
    own_time = observation["team_state"][0]["time_to_ball_ms"]
    opponent_time = observation["team_state"][1]["time_to_ball_ms"]
    time_to_ball = 0.5 + 0.5 * _clamp((opponent_time - own_time) / 2000.0, -1.0, 1.0)
    return (
        0.25 * possession_score
        + 0.20 * time_to_ball
        + 0.20 * _passing_options(observation)
        + 0.15 * _local_superiority(observation)
        + 0.10 * _team_structure(observation)
        + 0.10 * _field_control(observation)
    )


def _passing_options(observation: Observation) -> float:
    if observation["ball_owned_team"] != 0:
        return 0.0
    owner = observation["ball_owned_player"]
    own = observation["teams"][0]
    opponents = [player for player in observation["teams"][1] if player["is_active"]]
    if not 0 <= owner < len(own) or not own[owner]["is_active"] or not opponents:
        return 0.0
    start = own[owner]["position"]
    qualities = []
    for index, teammate in enumerate(own):
        if index == owner or not teammate["is_active"]:
            continue
        target = teammate["position"]
        receiver_space = min(
            _distance(target, opponent["position"]) for opponent in opponents
        )
        lane_space = min(
            _segment_distance(opponent["position"], start, target)
            for opponent in opponents
        )
        distance = _distance(start, target)
        reachability = math.exp(-abs(distance - 0.3) / 0.3)
        qualities.append(
            0.45 * _clamp(receiver_space / 0.20)
            + 0.45 * _clamp(lane_space / 0.08)
            + 0.10 * reachability
        )
    qualities.sort(reverse=True)
    return sum(qualities[:3]) / 3.0 if qualities else 0.0


def _local_superiority(observation: Observation) -> float:
    ball = observation["ball_position"]
    counts = []
    for team in observation["teams"]:
        counts.append(
            sum(
                player["is_active"] and _distance(player["position"], ball) <= 0.25
                for player in team
            )
        )
    return _clamp(0.5 + (counts[0] - counts[1]) / 6.0)


def _team_structure(observation: Observation) -> float:
    positions = [
        player["position"]
        for player in observation["teams"][0]
        if player["is_active"]
    ]
    if len(positions) < 2:
        return 0.0
    if observation["ball_owned_team"] == 0:
        width = max(position[1] for position in positions) - min(
            position[1] for position in positions
        )
        depth = max(position[0] for position in positions) - min(
            position[0] for position in positions
        )
        return 0.5 * _clamp(width / 0.65) + 0.5 * _clamp(depth / 1.2)
    center_x = sum(position[0] for position in positions) / len(positions)
    center_y = sum(position[1] for position in positions) / len(positions)
    mean_radius = sum(
        math.hypot(position[0] - center_x, position[1] - center_y)
        for position in positions
    ) / len(positions)
    return math.exp(-abs(mean_radius - 0.30) / 0.20)


def _field_control(observation: Observation) -> float:
    x = _clamp((observation["ball_position"][0] + 1.0) * 0.5)
    centrality = 1.0 - _clamp(abs(observation["ball_position"][1]) / 0.42)
    if observation["ball_owned_team"] == 0:
        return x * (0.7 + 0.3 * centrality)
    if observation["ball_owned_team"] == 1:
        danger = (1.0 - x) * (0.7 + 0.3 * centrality)
        return 1.0 - danger
    return 0.5


def _distance(first: list[float], second: list[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _segment_distance(point: list[float], start: list[float], end: list[float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return _distance(point, start)
    projection = _clamp(
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
        / length_squared
    )
    closest = [start[0] + projection * dx, start[1] + projection * dy]
    return _distance(point, closest)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
