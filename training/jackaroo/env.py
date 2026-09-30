"""Python wrappers for the native single-agent football environments."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
if os.name == "nt" and BIN.is_dir():
    os.add_dll_directory(str(BIN))

from . import _gfootball_env as native


Observation = dict[str, Any]


def _goal_result(previous: Observation, current: Observation) -> int:
    previous_difference = int(previous["goals"][0]) - int(previous["goals"][1])
    current_difference = int(current["goals"][0]) - int(current["goals"][1])
    result = current_difference - previous_difference
    if result not in (-1, 0, 1):
        raise RuntimeError("one engine step changed the score by more than one goal")
    return result


def _native_paths() -> tuple[Path, Path]:
    data = BIN / "data" if (BIN / "data").is_dir() else ROOT / "engine" / "data"
    fonts = BIN / "fonts" if (BIN / "fonts").is_dir() else ROOT / "engine" / "fonts"
    return data, fonts / "AlegreyaSansSC-ExtraBold.ttf"


class FootballEnv:
    """One external agent playing either physical side against built-in AI."""

    action_count = native.ENGINE_ACTION_COUNT

    def __init__(self, maximum_steps: int = 3000, seed: int | None = None) -> None:
        data, font = _native_paths()
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._next_left_team = True
        self._native = native.create(str(data), str(font), maximum_steps)
        self._observation: Observation | None = None
        self._segment_action_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._seed_generator.getrandbits(32)
        left_team = self._next_left_team
        self._next_left_team = not self._next_left_team
        self._observation = native.reset(
            self._native, seed & 0xFFFFFFFF, left_team
        )
        self._segment_action_count = 0
        return self._observation

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Advance one engine step and report segment and match boundaries."""

        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count - 1}]")
        previous = self._observation
        action_applied = bool(previous["is_in_play"])
        decision = [native.ENGINE_ACTION_COUNT] * 11
        player = previous["team_state"][0]["designated_possession_player"]
        if 0 <= player < 11 and previous["teams"][0][player]["is_active"]:
            decision[player] = action
        current, match_done = native.step(self._native, decision)
        if action_applied:
            self._segment_action_count += 1
        result = _goal_result(previous, current)
        segment_done = result != 0 or match_done
        self._observation = current
        info = {
            "goals": current["goals"],
            "step": current["step"],
            "segment_result": result if segment_done else None,
            "segment_steps": self._segment_action_count if segment_done else 0,
            "match_done": match_done,
            "action_applied": action_applied,
        }
        if segment_done:
            self._segment_action_count = 0
        return current, float(result), segment_done, info


class VectorFootballEnv:
    """Concurrent headless matches with Python-side reward calculation."""

    action_count = native.ENGINE_ACTION_COUNT

    def __init__(
        self,
        count: int,
        maximum_steps: int = 3000,
        seed: int | None = None,
        fixed_seed: int | None = None,
    ) -> None:
        if count <= 0:
            raise ValueError("environment count must be positive")
        data, font = _native_paths()
        self.count = count
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._fixed_seed = (
            None if fixed_seed is None else int(fixed_seed) & 0xFFFFFFFF
        )
        self._used_seeds: set[int] = set()
        self._next_left_team = [index % 2 == 0 for index in range(count)]
        self._native = native.create_batch(str(data), str(font), maximum_steps, count)
        self._observations: list[Observation] = []
        self._segment_action_counts = [0] * count

    def _seed(self) -> int:
        if self._fixed_seed is not None:
            return self._fixed_seed
        while True:
            seed = self._seed_generator.getrandbits(32)
            if seed not in self._used_seeds:
                self._used_seeds.add(seed)
                return seed

    def _take_side(self, index: int) -> bool:
        left_team = self._next_left_team[index]
        self._next_left_team[index] = not left_team
        return left_team

    def reset(self) -> list[Observation]:
        seeds = [self._seed() for _ in range(self.count)]
        sides = [self._take_side(index) for index in range(self.count)]
        self._observations = list(native.reset_batch(self._native, seeds, sides))
        for index, observation in enumerate(self._observations):
            self._segment_action_counts[index] = 0
        return self._observations

    def reset_one(self, index: int) -> Observation:
        if not 0 <= index < self.count:
            raise IndexError("environment index is out of range")
        observation = native.reset_batch_one(
            self._native, index, self._seed(), self._take_side(index)
        )
        self._observations[index] = observation
        self._segment_action_counts[index] = 0
        return observation

    def step(
        self, actions: Sequence[int], active: Sequence[bool] | None = None
    ) -> tuple[
        list[Observation], list[float], list[bool], list[dict[str, Any]]
    ]:
        """Advance active matches and leave completed segment slots frozen."""

        if len(actions) != self.count:
            raise ValueError("one action is required per environment")
        active_flags = [True] * self.count if active is None else list(active)
        if len(active_flags) != self.count:
            raise ValueError("one activity flag is required per environment")
        decisions = []
        previous_observations = self._observations
        for index, action in enumerate(actions):
            if not 0 <= action < self.action_count:
                raise ValueError(f"action must be in [0, {self.action_count - 1}]")
            decision = [native.ENGINE_ACTION_COUNT] * 11
            previous = previous_observations[index]
            player = previous["team_state"][0]["designated_possession_player"]
            if 0 <= player < 11 and previous["teams"][0][player]["is_active"]:
                decision[player] = action
            decisions.append(decision)

        results = native.step_batch(self._native, decisions, active_flags)
        observations = [result[0] for result in results]
        match_done = [bool(result[1]) for result in results]
        rewards: list[float] = []
        segment_done: list[bool] = []
        infos: list[dict[str, Any]] = []
        for index, current in enumerate(observations):
            if not active_flags[index]:
                rewards.append(0.0)
                segment_done.append(False)
                infos.append(
                    {
                        "goals": current["goals"],
                        "step": current["step"],
                        "segment_result": None,
                        "segment_steps": 0,
                        "match_done": match_done[index],
                        "action_applied": False,
                    }
                )
                continue
            result = _goal_result(previous_observations[index], current)
            action_applied = bool(previous_observations[index]["is_in_play"])
            if action_applied:
                self._segment_action_counts[index] += 1
            ended = result != 0 or match_done[index]
            rewards.append(float(result))
            segment_done.append(ended)
            infos.append(
                {
                    "goals": current["goals"],
                    "step": current["step"],
                    "segment_result": result if ended else None,
                    "segment_steps": (
                        self._segment_action_counts[index] if ended else 0
                    ),
                    "match_done": match_done[index],
                    "action_applied": action_applied,
                }
            )
            if ended:
                self._segment_action_counts[index] = 0
        self._observations = observations
        return observations, rewards, segment_done, infos
