"""Mapping from Jackaroo policy actions to model-interface decisions."""

from __future__ import annotations

from typing import Any


Observation = dict[str, Any]
Decision = tuple[int, ...]
ENGINE_ACTION_COUNT = 32
DIRECTED_ACTION_COUNT = 32
POLICY_ACTION_COUNT = ENGINE_ACTION_COUNT + DIRECTED_ACTION_COUNT
PLAYER_COUNT = 11
DELEGATE = 32


class JackarooAdapter:
    """Expand two-step actions while preserving the original recipient."""

    def __init__(self) -> None:
        self._pending: tuple[int, int] | None = None

    def reset(self) -> None:
        self._pending = None

    def action_required(self, observation: Observation) -> bool:
        """Return whether this step needs inference for its designated player."""

        pending = self._active_pending(observation)
        designated = _designated_player(observation)
        return designated >= 0 and (
            pending is None or pending[0] != designated
        )

    def map(self, observation: Observation, action: int | None) -> Decision:
        """Build one complete decision for exactly one engine step."""

        pending = self._active_pending(observation)
        designated = _designated_player(observation)
        requires_action = designated >= 0 and (
            pending is None or pending[0] != designated
        )
        if requires_action != (action is not None):
            expected = "a policy action" if requires_action else "no policy action"
            raise ValueError(f"this engine step requires {expected}")
        if action is not None and not 0 <= action < POLICY_ACTION_COUNT:
            raise ValueError(f"action must be in [0, {POLICY_ACTION_COUNT - 1}]")

        self._pending = None
        actions: list[tuple[int, int]] = []
        if pending is not None:
            actions.append(pending)
        if action is not None:
            if action < ENGINE_ACTION_COUNT:
                actions.append((designated, action))
            else:
                directed = action - ENGINE_ACTION_COUNT
                actions.append((designated, 9 + directed // 8))
                self._pending = (designated, 1 + directed % 8)
        return _decision(*actions)

    def _active_pending(self, observation: Observation) -> tuple[int, int] | None:
        if self._pending is not None and _is_active(observation, self._pending[0]):
            return self._pending
        self._pending = None
        return None


def _decision(*actions: tuple[int, int]) -> Decision:
    decision = [DELEGATE] * PLAYER_COUNT
    for player, action in actions:
        decision[player] = action
    return tuple(decision)


def _is_active(observation: Observation, player: int) -> bool:
    return 0 <= player < PLAYER_COUNT and observation["teams"][0][player]["is_active"]


def _designated_player(observation: Observation) -> int:
    player = observation["team_state"][0]["designated_possession_player"]
    return player if _is_active(observation, player) else -1
