"""ODA entity encoding and relational attention for Jackaroo."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .features import encode


Observation = dict[str, Any]
PLAYER_FEATURES = 24
CONTEXT_FEATURES = 66
BALL_FEATURES = 9
MATCH_FEATURES = CONTEXT_FEATURES - BALL_FEATURES
ENTITY_WIDTH = 64
RELATION_WIDTH = 64


def player_features(player: dict[str, Any]) -> list[float]:
    """Convert one public player record into a relational feature vector."""

    values: list[float] = []
    values.extend(player["position"])
    values.extend(player["velocity"])
    values.extend(player["facing"])
    values.extend(player["formation_position"])
    values.extend(player["dynamic_formation_position"])
    values.extend(
        (
            player["tired_factor"],
            player["role"] / 10.0,
            player["dynamic_role"] / 10.0,
            player["function_type"] / 14.0,
            player["action_frame"] / 1000.0,
            player["touch_frame"] / 1000.0,
            player["possession_duration_ms"] / 10000.0,
            player["time_to_ball_ms"] / 10000.0,
            float(player["has_card"]),
            float(player["is_active"]),
            float(player["touch_pending"]),
        )
    )
    return values


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def context_features(observation: Observation, maximum_steps: int) -> list[float]:
    """Encode ball, score, time, possession, and match mode context."""

    values: list[float] = []
    values.extend(observation["ball_position"])
    values.extend(observation["ball_velocity"])
    values.extend(observation["ball_rotation"])
    values.extend(goal / 5.0 for goal in observation["goals"])
    values.extend(_one_hot(observation["game_mode"], 7))
    values.extend(_one_hot(observation["set_piece_team"] + 1, 3))
    values.extend(_one_hot(observation["set_piece_taker"] + 1, 12))
    values.extend(_one_hot(observation["ball_owned_team"] + 1, 3))
    values.extend(_one_hot(observation["ball_owned_player"] + 1, 12))
    values.extend(_one_hot(observation["last_touch_team"] + 1, 3))
    values.extend(_one_hot(observation["last_touch_player"] + 1, 12))
    values.extend(
        (
            observation["match_time_ms"] / 6000000.0,
            observation["step"] / max(maximum_steps, 1),
            float(observation["is_in_play"]),
        )
    )
    return values


def padded_history(
    observations: Sequence[Observation], length: int
) -> list[Observation]:
    """Return one fixed causal history, padding with its first observation."""

    if not observations:
        raise ValueError("observation history must contain at least one state")
    if length <= 0:
        raise ValueError("history length must be positive")
    window = list(observations[-length:])
    return [window[0]] * (length - len(window)) + window


def batch_histories(
    histories: Sequence[Sequence[Observation]],
    maximum_steps: int,
    history_length: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build global, entity, context, and anchor tensors for causal histories."""

    windows = [padded_history(history, history_length) for history in histories]
    encoded = torch.stack(
        [torch.stack([encode(observation, maximum_steps) for observation in window])
         for window in windows]
    ).to(device)
    own = torch.tensor(
        [[[player_features(player) for player in observation["teams"][0]]
          for observation in window] for window in windows],
        dtype=torch.float32,
        device=device,
    )
    opponent = torch.tensor(
        [[[player_features(player) for player in observation["teams"][1]]
          for observation in window] for window in windows],
        dtype=torch.float32,
        device=device,
    )
    context = torch.tensor(
        [[context_features(observation, maximum_steps) for observation in window]
         for window in windows],
        dtype=torch.float32,
        device=device,
    )
    anchors = []
    opponent_anchors = []
    for window in windows:
        window_anchors = []
        window_opponent_anchors = []
        for observation in window:
            designated = observation["team_state"][0][
                "designated_possession_player"
            ]
            if not 0 <= designated < 11:
                designated = observation["ball_owned_player"]
            window_anchors.append(designated if 0 <= designated < 11 else 0)
            opponent_designated = observation["team_state"][1][
                "designated_possession_player"
            ]
            if not 0 <= opponent_designated < 11:
                opponent_designated = (
                    observation["ball_owned_player"]
                    if observation["ball_owned_team"] == 1
                    else 0
                )
            window_opponent_anchors.append(
                opponent_designated if 0 <= opponent_designated < 11 else 0
            )
        anchors.append(window_anchors)
        opponent_anchors.append(window_opponent_anchors)
    anchor_indices = torch.tensor(anchors, dtype=torch.long, device=device)
    opponent_anchor_indices = torch.tensor(
        opponent_anchors, dtype=torch.long, device=device
    )
    return (
        encoded,
        own,
        opponent,
        context,
        anchor_indices,
        opponent_anchor_indices,
    )


class ODAEncoder(nn.Module):
    """Relate every player to opponents, teammates, ball, and match context."""

    output_width = RELATION_WIDTH * 3

    def __init__(self, entity_width: int = ENTITY_WIDTH) -> None:
        super().__init__()
        self.entity = nn.Sequential(
            nn.Linear(PLAYER_FEATURES, entity_width),
            nn.Tanh(),
            nn.Linear(entity_width, entity_width),
            nn.Tanh(),
        )
        self.ball = nn.Sequential(
            nn.Linear(BALL_FEATURES, entity_width),
            nn.Tanh(),
        )
        self.match = nn.Sequential(
            nn.Linear(MATCH_FEATURES, entity_width),
            nn.Tanh(),
        )
        self.context = nn.Sequential(
            nn.Linear(entity_width * 2, entity_width),
            nn.Tanh(),
        )
        self.query_opponent = nn.Linear(entity_width * 2, entity_width, bias=False)
        self.key_opponent = nn.Linear(entity_width, entity_width, bias=False)
        self.value_opponent = nn.Linear(entity_width, entity_width, bias=False)
        self.relation = nn.Sequential(
            nn.Linear(entity_width * 3, RELATION_WIDTH),
            nn.Tanh(),
        )
        self.offense_query = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)
        self.offense_key = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)
        self.defense_query = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)
        self.defense_key = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)
        self.offense_value = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)
        self.defense_value = nn.Linear(RELATION_WIDTH, RELATION_WIDTH, bias=False)

    def _relations(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_opponent(
            torch.cat((primary, context.unsqueeze(1).expand_as(primary)), dim=-1)
        )
        keys = self.key_opponent(secondary)
        scores = torch.einsum("bid,bjd->bij", query, keys) / keys.shape[-1] ** 0.5
        active = secondary[..., -2] > 0.5
        active = torch.where(
            active.any(dim=-1, keepdim=True), active, torch.ones_like(active)
        )
        scores = scores.masked_fill(~active.unsqueeze(1), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        opponent_context = torch.einsum(
            "bij,bjd->bid", weights, self.value_opponent(secondary)
        )
        expanded_context = context.unsqueeze(1).expand_as(primary)
        return self.relation(
            torch.cat((primary, opponent_context, expanded_context), dim=-1)
        )

    def forward(
        self,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = own.shape[:-2]
        own_flat = own.reshape(-1, 11, PLAYER_FEATURES)
        opponent_flat = opponent.reshape(-1, 11, PLAYER_FEATURES)
        context_flat = context.reshape(-1, CONTEXT_FEATURES)
        anchors_flat = anchor_indices.reshape(-1)
        opponent_anchors_flat = opponent_anchor_indices.reshape(-1)
        own_encoded = self.entity(own_flat)
        opponent_encoded = self.entity(opponent_flat)
        context_encoded = self.context(
            torch.cat(
                (
                    self.ball(context_flat[:, :BALL_FEATURES]),
                    self.match(context_flat[:, BALL_FEATURES:]),
                ),
                dim=-1,
            )
        )
        own_relation = self._relations(own_encoded, opponent_encoded, context_encoded)
        opponent_relation = self._relations(
            opponent_encoded, own_encoded, context_encoded
        )
        rows = torch.arange(own_flat.shape[0], device=own.device)
        anchor = own_relation[rows, anchors_flat]
        opponent_anchor = opponent_relation[rows, opponent_anchors_flat]

        offense_logits = torch.einsum(
            "bd,bid->bi",
            self.offense_query(anchor),
            self.offense_key(own_relation),
        ) / RELATION_WIDTH ** 0.5
        defense_logits = torch.einsum(
            "bd,bid->bi",
            self.defense_query(opponent_anchor),
            self.defense_key(opponent_relation),
        ) / RELATION_WIDTH ** 0.5
        own_active = own_flat[..., -2] > 0.5
        opponent_active = opponent_flat[..., -2] > 0.5
        own_active = torch.where(
            own_active.any(dim=-1, keepdim=True),
            own_active,
            torch.ones_like(own_active),
        )
        opponent_active = torch.where(
            opponent_active.any(dim=-1, keepdim=True),
            opponent_active,
            torch.ones_like(opponent_active),
        )
        offense_logits = offense_logits.masked_fill(~own_active, -torch.inf)
        defense_logits = defense_logits.masked_fill(~opponent_active, -torch.inf)
        offense_weights = torch.softmax(offense_logits, dim=-1)
        defense_weights = torch.softmax(defense_logits, dim=-1)
        teammate_context = torch.einsum(
            "bi,bid->bd", offense_weights, self.offense_value(own_relation)
        )
        opponent_context = torch.einsum(
            "bi,bid->bd", defense_weights, self.defense_value(opponent_relation)
        )
        relation = torch.cat((anchor, teammate_context, opponent_context), dim=-1)
        return (
            relation.reshape(*shape, self.output_width),
            offense_logits.reshape(*shape, 11),
            defense_logits.reshape(*shape, 11),
        )
