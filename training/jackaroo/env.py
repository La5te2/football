"""Python wrappers for the native single-agent football environments."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
from typing import Any, Sequence

from .reward import football_potential, transition_reward


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
if os.name == "nt" and BIN.is_dir():
    os.add_dll_directory(str(BIN))

from . import _gfootball_env as native


Observation = dict[str, Any]
POTENTIAL_DISCOUNT = 1.0
POTENTIAL_SCALE = 0.20


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
        self._potential_scale = POTENTIAL_SCALE
        self._potential_cache = 0.0
        self._initial_scaled_potential = 0.0
        self._potential_reward_total = 0.0

    def set_potential_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("potential scale must be nonnegative")
        self._potential_scale = scale
        if self._observation is not None and self._observation["step"] == 0:
            self._initial_scaled_potential = scale * self._potential_cache

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._seed_generator.getrandbits(32)
        left_team = self._next_left_team
        self._next_left_team = not self._next_left_team
        self._observation = native.reset(
            self._native, seed & 0xFFFFFFFF, left_team
        )
        self._potential_cache = football_potential(self._observation)
        self._initial_scaled_potential = self._potential_scale * self._potential_cache
        self._potential_reward_total = 0.0
        return self._observation

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count - 1}]")
        previous = self._observation
        decision = [native.ENGINE_ACTION_COUNT] * 11
        player = previous["team_state"][0]["designated_possession_player"]
        if 0 <= player < 11 and previous["teams"][0][player]["is_active"]:
            decision[player] = action
        current, terminated = native.step(self._native, decision)
        reward, self._potential_cache, components = transition_reward(
            previous,
            current,
            terminated,
            self._potential_cache,
            self._potential_scale,
            POTENTIAL_DISCOUNT,
        )
        if not math.isfinite(reward):
            raise FloatingPointError("environment reward is not finite")
        self._potential_reward_total += components["potential"]
        self._observation = current
        info = {
            "goals": current["goals"],
            "step": current["step"],
            "reward_components": components,
            "potential_telescoping_error": (
                self._potential_reward_total + self._initial_scaled_potential
                if terminated
                else 0.0
            ),
        }
        return current, reward, terminated, info


class VectorFootballEnv:
    """Concurrent headless matches with Python-side reward calculation."""

    action_count = native.ENGINE_ACTION_COUNT

    def __init__(
        self, count: int, maximum_steps: int = 3000, seed: int | None = None
    ) -> None:
        if count <= 0:
            raise ValueError("environment count must be positive")
        data, font = _native_paths()
        self.count = count
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._used_seeds: set[int] = set()
        self._next_left_team = [index % 2 == 0 for index in range(count)]
        self._native = native.create_batch(str(data), str(font), maximum_steps, count)
        self._observations: list[Observation] = []
        self._potential_scale = POTENTIAL_SCALE
        self._potential_cache = [0.0] * count
        self._initial_scaled_potential = [0.0] * count
        self._potential_reward_total = [0.0] * count

    def _seed(self) -> int:
        while True:
            seed = self._seed_generator.getrandbits(32)
            if seed not in self._used_seeds:
                self._used_seeds.add(seed)
                return seed

    def _take_side(self, index: int) -> bool:
        left_team = self._next_left_team[index]
        self._next_left_team[index] = not left_team
        return left_team

    def set_potential_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("potential scale must be nonnegative")
        self._potential_scale = scale

    def _initialize_reward(self, index: int, observation: Observation) -> None:
        value = football_potential(observation)
        self._potential_cache[index] = value
        self._initial_scaled_potential[index] = self._potential_scale * value
        self._potential_reward_total[index] = 0.0

    def reset(self) -> list[Observation]:
        seeds = [self._seed() for _ in range(self.count)]
        sides = [self._take_side(index) for index in range(self.count)]
        self._observations = list(native.reset_batch(self._native, seeds, sides))
        for index, observation in enumerate(self._observations):
            self._initialize_reward(index, observation)
        return self._observations

    def reset_builtin(
        self, seeds: Sequence[int], sides: Sequence[bool]
    ) -> list[Observation]:
        """Reset pure built-in matches with explicit reproducible seeds."""

        if len(seeds) != self.count or len(sides) != self.count:
            raise ValueError("one seed and side are required per environment")
        self._observations = list(
            native.reset_builtin_batch(
                self._native,
                [int(seed) & 0xFFFFFFFF for seed in seeds],
                list(sides),
            )
        )
        return self._observations

    def reset_one(self, index: int) -> Observation:
        if not 0 <= index < self.count:
            raise IndexError("environment index is out of range")
        observation = native.reset_batch_one(
            self._native, index, self._seed(), self._take_side(index)
        )
        self._observations[index] = observation
        self._initialize_reward(index, observation)
        return observation

    def step_builtin(self) -> tuple[list[Observation], list[bool]]:
        """Advance pure built-in matches through the threaded native batch."""

        decisions = [[native.ENGINE_ACTION_COUNT] * 11 for _ in range(self.count)]
        results = native.step_batch(self._native, decisions)
        self._observations = [result[0] for result in results]
        return self._observations, [bool(result[1]) for result in results]

    def step(self, actions: Sequence[int]) -> tuple[
        list[Observation], list[float], list[bool], list[dict[str, Any]]
    ]:
        if len(actions) != self.count:
            raise ValueError("one action is required per environment")
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

        results = native.step_batch(self._native, decisions)
        observations = [result[0] for result in results]
        terminated = [bool(result[1]) for result in results]
        rewards: list[float] = []
        infos: list[dict[str, Any]] = []
        for index, current in enumerate(observations):
            reward, self._potential_cache[index], components = transition_reward(
                previous_observations[index],
                current,
                terminated[index],
                self._potential_cache[index],
                self._potential_scale,
                POTENTIAL_DISCOUNT,
            )
            if not math.isfinite(reward):
                raise FloatingPointError("environment reward is not finite")
            self._potential_reward_total[index] += components["potential"]
            rewards.append(reward)
            infos.append(
                {
                    "goals": current["goals"],
                    "step": current["step"],
                    "reward_components": components,
                    "potential_telescoping_error": (
                        self._potential_reward_total[index]
                        + self._initial_scaled_potential[index]
                        if terminated[index]
                        else 0.0
                    ),
                }
            )
        self._observations = observations
        return observations, rewards, terminated, infos
