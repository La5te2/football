"""Python environment wrapper and task reward for single-agent training."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "bin"
if os.name == "nt" and BIN.is_dir():
    os.add_dll_directory(str(BIN))

from . import _gfootball_env as native
from .adapter import JackarooAdapter, POLICY_ACTION_COUNT


Observation = dict[str, Any]


class FootballEnv:
    """One left-side external agent playing against the built-in team AI."""

    action_count = POLICY_ACTION_COUNT

    def __init__(self, maximum_steps: int = 3000, seed: int | None = None) -> None:
        data = BIN / "data" if (BIN / "data").is_dir() else ROOT / "engine" / "data"
        fonts = BIN / "fonts" if (BIN / "fonts").is_dir() else ROOT / "engine" / "fonts"
        font = fonts / "AlegreyaSansSC-ExtraBold.ttf"
        self.maximum_steps = maximum_steps
        self._seed_generator = random.Random(seed)
        self._adapter = JackarooAdapter()
        self._native = native.create(str(data), str(font), maximum_steps)
        self._observation: Observation | None = None

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._seed_generator.getrandbits(32)
        self._adapter.reset()
        self._observation = native.reset(self._native, seed & 0xFFFFFFFF)
        return self._observation

    @property
    def action_required(self) -> bool:
        if self._observation is None:
            raise RuntimeError("reset() must be called before reading action_required")
        return self._adapter.action_required(self._observation)

    def step(
        self, action: int | None
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        previous = self._observation
        decision = self._adapter.map(previous, action)
        current, terminated = native.step(self._native, decision)
        reward = self._reward(previous, current)
        self._observation = current
        info = {
            "goals": current["goals"],
            "step": current["step"],
        }
        return current, reward, terminated, info

    @staticmethod
    def _reward(previous: Observation, current: Observation) -> float:
        own_goal = current["goals"][0] - previous["goals"][0]
        opponent_goal = current["goals"][1] - previous["goals"][1]
        score = float(own_goal - opponent_goal)
        if score != 0.0:
            return score

        progress = 0.05 * (
            current["ball_position"][0] - previous["ball_position"][0]
        )
        possession_change = 0.0
        if previous["ball_owned_team"] != current["ball_owned_team"]:
            if current["ball_owned_team"] == 0:
                possession_change = 0.01
            elif previous["ball_owned_team"] == 0:
                possession_change = -0.01
        return progress + possession_change
