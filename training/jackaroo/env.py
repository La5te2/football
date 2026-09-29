"""Python environment wrapper and task reward for single-agent training."""

from __future__ import annotations

import os
import math
from pathlib import Path
import random
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
if os.name == "nt" and BIN.is_dir():
    os.add_dll_directory(str(BIN))

from . import _gfootball_env as native


Observation = dict[str, Any]
PotentialFunction = Callable[[Sequence[Observation]], float]
BatchPotentialFunction = Callable[
    [Sequence[Sequence[Observation]]], Sequence[float]
]

POTENTIAL_DISCOUNT = 1.0
POTENTIAL_SCALE = 0.02


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
        self._potential_fn: PotentialFunction | None = None
        self._potential_scale = POTENTIAL_SCALE
        self._potential_history: list[Observation] = []
        self._potential_cache = 0.0
        self._initial_scaled_potential = 0.0
        self._potential_reward_total = 0.0

    def set_potential(
        self, potential_fn: PotentialFunction | None
    ) -> None:
        """Install the frozen potential used by the next PPO interval."""

        self._potential_fn = potential_fn
        if self._observation is not None:
            self._potential_cache = (
                potential_fn(self._potential_history) if potential_fn else 0.0
            )
            if self._observation["step"] == 0:
                self._initial_scaled_potential = (
                    self._potential_scale * self._potential_cache
                )
                self._potential_reward_total = 0.0

    def set_potential_scale(self, scale: float) -> None:
        """Set the optimization scale applied to potential differences."""

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
        self._potential_history = [self._observation]
        self._potential_cache = (
            self._potential_fn(self._potential_history)
            if self._potential_fn
            else 0.0
        )
        self._initial_scaled_potential = (
            self._potential_scale * self._potential_cache
        )
        self._potential_reward_total = 0.0
        return self._observation

    def collect_builtin_episode(
        self,
        seed: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[Observation], tuple[int, int]]:
        """Collect one complete all-delegated match for potential pretraining."""

        observation = self.reset(seed)
        observations = [observation]
        next_progress_step = 500
        while True:
            decision = [self.action_count] * 11
            observation, terminated = native.step(self._native, decision)
            observations.append(observation)
            self._observation = observation
            reached_progress_step = observation["step"] >= next_progress_step
            if progress is not None and (reached_progress_step or terminated):
                progress(observation["step"], self.maximum_steps)
            if reached_progress_step:
                next_progress_step = observation["step"] + 500
            if terminated:
                return observations, tuple(observation["goals"])

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
        current, terminated = native.step(self._native, decision)
        reward, reward_components = self._reward(previous, current, terminated)
        if not math.isfinite(reward):
            raise FloatingPointError("environment reward is not finite")
        self._observation = current
        self._potential_history.append(current)
        info = {
            "goals": current["goals"],
            "step": current["step"],
            "reward_components": reward_components,
            "potential_telescoping_error": (
                self._potential_reward_total + self._initial_scaled_potential
                if terminated
                else 0.0
            ),
        }
        return current, reward, terminated, info

    def _reward(
        self, previous: Observation, current: Observation, terminated: bool
    ) -> tuple[float, dict[str, float]]:
        score = 0.0
        if terminated:
            own_goals, opponent_goals = current["goals"]
            score = float((own_goals > opponent_goals) - (own_goals < opponent_goals))
        previous_potential = self._potential_cache
        current_potential = (
            0.0
            if terminated or self._potential_fn is None
            else self._potential_fn(self._potential_history + [current])
        )
        potential = self._potential_scale * (
            POTENTIAL_DISCOUNT * current_potential - previous_potential
        )
        self._potential_reward_total += potential
        self._potential_cache = current_potential
        components = {
            "score": score,
            "potential": potential,
        }
        return sum(components.values()), components


class VectorFootballEnv:
    """Concurrent headless matches with batched potential evaluation."""

    action_count = native.ENGINE_ACTION_COUNT

    def __init__(
        self, count: int, maximum_steps: int = 3000, seed: int | None = None
    ) -> None:
        if count <= 0:
            raise ValueError("environment count must be positive")
        data = BIN / "data" if (BIN / "data").is_dir() else ROOT / "engine" / "data"
        fonts = BIN / "fonts" if (BIN / "fonts").is_dir() else ROOT / "engine" / "fonts"
        font = fonts / "AlegreyaSansSC-ExtraBold.ttf"
        self.count = count
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._used_seeds: set[int] = set()
        self._next_left_team = [index % 2 == 0 for index in range(count)]
        self._native = native.create_batch(
            str(data), str(font), maximum_steps, count
        )
        self._observations: list[Observation] = []
        self._next_potential_fn: BatchPotentialFunction | None = None
        self._potential_fns: list[BatchPotentialFunction | None] = [
            None for _ in range(count)
        ]
        self._potential_scale = POTENTIAL_SCALE
        self._potential_scales = [POTENTIAL_SCALE] * count
        self._potential_histories: list[list[Observation]] = [
            [] for _ in range(count)
        ]
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

    def set_potential(
        self, potential_fn: BatchPotentialFunction | None
    ) -> None:
        """Use a new frozen potential when each environment starts its next match."""

        self._next_potential_fn = potential_fn
        if not self._observations:
            self._potential_fns = [potential_fn for _ in range(self.count)]

    def set_potential_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("potential scale must be nonnegative")
        self._potential_scale = scale
        if not self._observations:
            self._potential_scales = [scale] * self.count

    def reset(self) -> list[Observation]:
        seeds = [self._seed() for _ in range(self.count)]
        sides = [self._take_side(index) for index in range(self.count)]
        self._observations = list(
            native.reset_batch(self._native, seeds, sides)
        )
        self._potential_fns = [self._next_potential_fn for _ in range(self.count)]
        self._potential_scales = [self._potential_scale] * self.count
        self._potential_histories = [
            [observation] for observation in self._observations
        ]
        self._potential_cache = (
            list(self._next_potential_fn(self._potential_histories))
            if self._next_potential_fn
            else [0.0] * self.count
        )
        if len(self._potential_cache) != self.count:
            raise ValueError("potential function returned the wrong batch size")
        self._initial_scaled_potential = [
            self._potential_scale * value for value in self._potential_cache
        ]
        self._potential_reward_total = [0.0] * self.count
        return self._observations

    def reset_one(self, index: int) -> Observation:
        if not 0 <= index < self.count:
            raise IndexError("environment index is out of range")
        observation = native.reset_batch_one(
            self._native, index, self._seed(), self._take_side(index)
        )
        self._observations[index] = observation
        self._potential_histories[index] = [observation]
        self._potential_fns[index] = self._next_potential_fn
        self._potential_scales[index] = self._potential_scale
        values = (
            self._next_potential_fn([[observation]])
            if self._next_potential_fn
            else [0.0]
        )
        if len(values) != 1:
            raise ValueError("potential function returned the wrong batch size")
        value = float(values[0])
        self._potential_cache[index] = value
        self._initial_scaled_potential[index] = self._potential_scale * value
        self._potential_reward_total[index] = 0.0
        return observation

    def step(
        self, actions: Sequence[int]
    ) -> tuple[
        list[Observation], list[float], list[bool], list[dict[str, Any]]
    ]:
        if len(actions) != self.count:
            raise ValueError("one action is required per environment")
        decisions = []
        for index, action in enumerate(actions):
            if not 0 <= action < self.action_count:
                raise ValueError(
                    f"action must be in [0, {self.action_count - 1}]"
                )
            decision = [native.ENGINE_ACTION_COUNT] * 11
            previous = self._observations[index]
            player = previous["team_state"][0]["designated_possession_player"]
            if 0 <= player < 11 and previous["teams"][0][player]["is_active"]:
                decision[player] = action
            decisions.append(decision)

        results = native.step_batch(self._native, decisions)
        observations = [result[0] for result in results]
        terminated = [bool(result[1]) for result in results]
        for index, current in enumerate(observations):
            self._potential_histories[index].append(current)
        current_potential = [0.0] * self.count
        active = [index for index, done in enumerate(terminated) if not done]
        potential_groups: dict[int, tuple[BatchPotentialFunction, list[int]]] = {}
        for index in active:
            function = self._potential_fns[index]
            if function is not None:
                potential_groups.setdefault(id(function), (function, []))[1].append(
                    index
                )
        for function, indices in potential_groups.values():
            values = function(
                [self._potential_histories[index] for index in indices]
            )
            if len(values) != len(indices):
                raise ValueError("potential function returned the wrong batch size")
            for index, value in zip(indices, values):
                current_potential[index] = float(value)

        rewards = []
        infos = []
        for index, current in enumerate(observations):
            own_goals, opponent_goals = current["goals"]
            score = (
                float((own_goals > opponent_goals) - (own_goals < opponent_goals))
                if terminated[index]
                else 0.0
            )
            potential = self._potential_scales[index] * (
                POTENTIAL_DISCOUNT * current_potential[index]
                - self._potential_cache[index]
            )
            reward = score + potential
            if not math.isfinite(reward):
                raise FloatingPointError("environment reward is not finite")
            self._potential_reward_total[index] += potential
            self._potential_cache[index] = current_potential[index]
            rewards.append(reward)
            infos.append(
                {
                    "goals": current["goals"],
                    "step": current["step"],
                    "reward_components": {
                        "score": score,
                        "potential": potential,
                    },
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
