"""TamakEri teacher inference for Jackaroo behavior pretraining."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

import torch


Observation = dict[str, Any]
TAMAKERI_WEIGHTS = Path(__file__).with_name("tamakeri.pt")
SELECTABLE_ACTIONS = 51
BUILTIN_AI_POLICY_ACTION = 19
DIRECTED_KICK_BEGIN = 20
DELEGATE = 32

POLICY_TO_ENGINE = (
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 18, 20, 30, 14, 19, 31,
)


@dataclass
class TeacherDecision:
    """One full team decision and its optional Jackaroo-compatible label."""

    actions: list[int]
    action: int | None


@dataclass
class _State:
    pending_action: int = -1
    pending_player: int = -1
    last_action: int = 0
    history: deque[int] = field(default_factory=deque)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _multiscale(value: float, scale: float) -> float:
    return 2.0 / (1.0 + math.exp(-value / scale))


def _designated(observation: Observation) -> int:
    player = int(observation["team_state"][0]["designated_possession_player"])
    if 0 <= player < 11 and observation["teams"][0][player]["is_active"]:
        return player
    return -1


def _features(
    observation: Observation, controlled_player: int, maximum_steps: int
) -> tuple[torch.Tensor, ...]:
    ball = tuple(float(value) for value in (
        *observation["ball_position"],
        *observation["ball_velocity"],
        *observation["ball_rotation"],
    ))
    own = observation["teams"][0]
    opponent = observation["teams"][1]
    ball_owned_team = int(observation["ball_owned_team"])
    ball_owned_player = int(observation["ball_owned_player"])
    game_mode = int(observation["game_mode"])
    if game_mode in (2, 3, 4, 6):
        best = float("inf")
        for side, team in enumerate((own, opponent)):
            for index, player in enumerate(team):
                distance = _distance(player["position"], ball)
                if distance < best:
                    best = distance
                    ball_owned_team = side
                    ball_owned_player = index

    goals = observation["goals"]
    match: list[float] = []
    for score in goals:
        match.extend((_multiscale(score, 1.0), _multiscale(score, 3.0)))
    difference = goals[0] - goals[1]
    match.extend((_multiscale(difference, 1.0), _multiscale(difference, 3.0)))
    steps_left = maximum_steps - int(observation["step"])
    for scale in (10.0, 100.0, 1000.0, 10000.0):
        match.append(_multiscale(steps_left, scale))
    half_left = steps_left - 1500 if steps_left > 1500 else steps_left
    for scale in (10.0, 100.0, 1000.0, 10000.0):
        match.append(_multiscale(half_left, scale))
    match.extend((float(ball_owned_team == 0), float(ball_owned_team == 1)))

    players: list[list[list[float]]] = [[], []]
    for side, team in enumerate((own, opponent)):
        for index, player in enumerate(team):
            players[side].append([
                float(side == 0),
                float(player["position"][0]),
                float(player["position"][1]),
                float(player["velocity"][0]),
                float(player["velocity"][1]),
                float(player["tired_factor"]),
                float(player["has_card"]),
                float(player["is_active"]),
                float(ball_owned_team == side and ball_owned_player == index),
            ])

    sticky = observation["sticky_actions"][controlled_player]
    direction_vectors = (
        (-1.0, 0.0), (-0.707, -0.707), (0.0, 1.0),
        (0.707, -0.707), (1.0, 0.0), (0.707, 0.707),
        (0.0, -1.0), (-0.707, 0.707),
    )
    control = [0.0, 0.0, float(sticky[8]), float(sticky[9])]
    for index, active in enumerate(sticky[:8]):
        if active:
            control[:2] = direction_vectors[index]
            break

    indices = list(range(11))
    control_flag = [0.0] * 22
    control_flag[controlled_player] = 1.0
    player_distances: list[list[float]] = []
    for team in (own, opponent):
        for player in team:
            position = player["position"]
            player_distances.append([
                _distance(position, ball),
                _distance(position, (-1.0, 0.0)),
                _distance(position, (1.0, 0.0)),
                abs(position[0] + 1.0), abs(position[0] - 1.0),
                abs(position[1] + 0.42), abs(position[1] - 0.42),
            ])
    ball_distances = [
        _distance(ball, (-1.0, 0.0)), _distance(ball, (1.0, 0.0)),
        abs(ball[0] + 1.0), abs(ball[0] - 1.0),
        abs(ball[1] + 0.42), abs(ball[1] - 0.42),
    ]

    own_position = torch.tensor(
        [player["position"][:2] for player in own], dtype=torch.float32
    )
    opponent_position = torch.tensor(
        [player["position"][:2] for player in opponent], dtype=torch.float32
    )
    own_velocity = torch.tensor(
        [player["velocity"][:2] for player in own], dtype=torch.float32
    )
    opponent_velocity = torch.tensor(
        [player["velocity"][:2] for player in opponent], dtype=torch.float32
    )
    left = own_position[:, None].expand(11, 11, 2)
    right = opponent_position[None].expand(11, 11, 2)
    left_velocity = own_velocity[:, None].expand(11, 11, 2)
    right_velocity = opponent_velocity[None].expand(11, 11, 2)
    active_position = own_position[controlled_player]
    def constant(value: float | torch.Tensor) -> torch.Tensor:
        return torch.full((11, 11), float(value))

    grid = torch.stack((
        left[..., 0], left[..., 1], right[..., 0], right[..., 1],
        constant(ball[0]), constant(ball[1]), constant(ball[2]),
        constant(-1.0), constant(0.0), constant(1.0), constant(0.0),
        constant(-0.42), constant(0.42),
        constant(active_position[0]), constant(active_position[1]),
        left[..., 0] - right[..., 0], left[..., 1] - right[..., 1],
        left[..., 0] - 1.0, left[..., 1], left[..., 0] + 1.0, left[..., 1],
        right[..., 0] - 1.0, right[..., 1],
        right[..., 0] + 1.0, right[..., 1],
        (left[..., 1] + 0.42).abs(), (left[..., 1] - 0.42).abs(),
        (right[..., 1] + 0.42).abs(), (right[..., 1] - 0.42).abs(),
        right[..., 0] - ball[0], right[..., 1] - ball[1],
        right[..., 0] - active_position[0],
        right[..., 1] - active_position[1],
        left[..., 0] - ball[0], left[..., 1] - ball[1],
        left[..., 0] - active_position[0], left[..., 1] - active_position[1],
        constant(ball[3]), constant(ball[4]), constant(ball[5]),
        left_velocity[..., 0] - ball[3], left_velocity[..., 1] - ball[4],
        right_velocity[..., 0] - ball[3], right_velocity[..., 1] - ball[4],
        left_velocity[..., 0], left_velocity[..., 1],
        right_velocity[..., 0], right_velocity[..., 1],
        left_velocity[..., 0] - right_velocity[..., 0],
        left_velocity[..., 1] - right_velocity[..., 1],
        constant(ball[6]), constant(ball[7]), constant(ball[8]),
    ))

    return (
        torch.tensor(ball, dtype=torch.float32).view(1, 9),
        torch.tensor(match, dtype=torch.float32).view(1, 16),
        torch.tensor(players[0], dtype=torch.float32).view(1, 11, 9),
        torch.tensor(players[1], dtype=torch.float32).view(1, 11, 9),
        torch.tensor(control, dtype=torch.float32).view(1, 4),
        torch.tensor(indices, dtype=torch.long).view(1, 11),
        torch.tensor(indices, dtype=torch.long).view(1, 11),
        torch.tensor([game_mode], dtype=torch.long).view(1, 1),
        torch.tensor(control_flag, dtype=torch.float32).view(1, 22, 1),
        torch.tensor(player_distances, dtype=torch.float32).view(1, 22, 7),
        torch.tensor(ball_distances, dtype=torch.float32).view(1, 6),
        grid.view(1, 53, 11, 11),
    )


class TamakEriTeacher:
    """Runs batched TamakEri inference with independent state per match side."""

    def __init__(
        self,
        agent_count: int,
        maximum_steps: int,
        device: torch.device,
        weights: Path = TAMAKERI_WEIGHTS,
    ) -> None:
        if agent_count <= 0:
            raise ValueError("teacher agent count must be positive")
        if not weights.is_file():
            raise FileNotFoundError(f"TamakEri weights not found: {weights}")
        self.device = device
        self.maximum_steps = maximum_steps
        self.model = torch.jit.load(str(weights), map_location=device).eval()
        self.states = [_State() for _ in range(agent_count)]

    def reset(self, indices: Sequence[int] | None = None) -> None:
        selected = range(len(self.states)) if indices is None else indices
        for index in selected:
            self.states[index] = _State()

    @staticmethod
    def _update_history(state: _State) -> None:
        state.history.append(state.last_action)
        while len(state.history) > 8:
            state.history.popleft()

    @staticmethod
    def _decision(player: int, action: int, designated: int) -> TeacherDecision:
        actions = [DELEGATE] * 11
        if 0 <= player < 11:
            actions[player] = action
        return TeacherDecision(
            actions,
            action if player == designated and 0 <= action < DELEGATE else None,
        )

    @staticmethod
    def _legal_action(observation: Observation, logits: torch.Tensor) -> int:
        sticky = observation["sticky_actions"][_designated(observation)]
        owns_ball = observation["ball_owned_team"] == 0
        best_action = 0
        best_value = float("-inf")
        values = logits.reshape(-1)
        for action in range(SELECTABLE_ACTIONS):
            legal = action != BUILTIN_AI_POLICY_ACTION
            if not owns_ball and (
                9 <= action <= 12 or action == 17 or action >= DIRECTED_KICK_BEGIN
            ):
                legal = False
            if owns_ball and action == 16:
                legal = False
            if not sticky[8] and action == 15:
                legal = False
            if not sticky[9] and action == 18:
                legal = False
            if not any(sticky[:8]) and action == 14:
                legal = False
            value = float(values[action])
            if legal and value > best_value:
                best_action = action
                best_value = value
        return best_action

    def decide(
        self, observations: Sequence[Observation], agent_indices: Sequence[int]
    ) -> list[TeacherDecision]:
        if len(observations) != len(agent_indices):
            raise ValueError("one teacher state index is required per observation")
        results: list[TeacherDecision | None] = [None] * len(observations)
        inference_rows: list[int] = []
        feature_rows: list[tuple[torch.Tensor, ...]] = []
        for row, (observation, agent_index) in enumerate(
            zip(observations, agent_indices)
        ):
            state = self.states[agent_index]
            designated = _designated(observation)
            if state.pending_action >= 0:
                self._update_history(state)
                action = state.pending_action
                player = state.pending_player
                state.pending_action = -1
                state.pending_player = -1
                if not (0 <= player < 11 and observation["teams"][0][player]["is_active"]):
                    player = -1
                else:
                    state.last_action = action
                results[row] = self._decision(player, action, designated)
                continue
            if designated < 0:
                results[row] = self._decision(-1, 0, designated)
                continue
            self._update_history(state)
            features = _features(observation, designated, self.maximum_steps)
            history = list(reversed(state.history))
            history.extend([0] * (8 - len(history)))
            history = history[:8]
            feature_rows.append(
                (*features, torch.tensor(history, dtype=torch.long).view(1, 8, 1))
            )
            inference_rows.append(row)

        if inference_rows:
            inputs = [
                torch.cat([features[index] for features in feature_rows]).to(
                    self.device, non_blocking=True
                )
                for index in range(13)
            ]
            with torch.inference_mode():
                logits = self.model(*inputs).cpu()
            for batch_row, row in enumerate(inference_rows):
                observation = observations[row]
                state = self.states[agent_indices[row]]
                designated = _designated(observation)
                policy_action = self._legal_action(observation, logits[batch_row])
                if policy_action >= DIRECTED_KICK_BEGIN:
                    offset = policy_action - DIRECTED_KICK_BEGIN
                    action = 9 + offset // 8
                    state.pending_action = offset % 8 + 1
                    state.pending_player = designated
                    state.last_action = action
                else:
                    action = POLICY_TO_ENGINE[policy_action]
                    state.last_action = policy_action
                results[row] = self._decision(designated, action, designated)

        if any(result is None for result in results):
            raise RuntimeError("teacher inference did not produce every decision")
        return [result for result in results if result is not None]
