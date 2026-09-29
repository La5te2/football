"""Tensor conversion and entity Transformer for the Jackaroo policy."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence, TypeVar

import torch
from torch import nn

from .features import encode


Observation = dict[str, Any]
BASE_PLAYER_FEATURES = 24
RELATIVE_PLAYER_FEATURES = 6
PLAYER_FEATURES = BASE_PLAYER_FEATURES + RELATIVE_PLAYER_FEATURES
PLAYER_ACTIVE_INDEX = BASE_PLAYER_FEATURES - 2
CONTEXT_FEATURES = 66
BALL_FEATURES = 9
MATCH_FEATURES = CONTEXT_FEATURES - BALL_FEATURES
ENTITY_WIDTH = 128
ENTITY_COUNT = 24
HistoryItem = TypeVar("HistoryItem")


class TensorFrame(NamedTuple):
    """One public observation encoded once in CPU tensors."""

    encoded: torch.Tensor
    own: torch.Tensor
    opponent: torch.Tensor
    context: torch.Tensor
    anchor: torch.Tensor
    opponent_anchor: torch.Tensor


def player_features(
    player: dict[str, Any],
    ball_position: Sequence[float],
    controlled_position: Sequence[float],
) -> list[float]:
    """Encode one player and its position relative to the ball and controller."""

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
    values.extend(
        coordinate - ball_coordinate
        for coordinate, ball_coordinate in zip(player["position"], ball_position)
    )
    values.extend(
        coordinate - controlled_coordinate
        for coordinate, controlled_coordinate in zip(
            player["position"], controlled_position
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
    observations: Sequence[HistoryItem], length: int
) -> list[HistoryItem]:
    """Return one fixed causal history, padding with its first observation."""

    if not observations:
        raise ValueError("observation history must contain at least one state")
    if length <= 0:
        raise ValueError("history length must be positive")
    window = list(observations[-length:])
    return [window[0]] * (length - len(window)) + window


def tensorize(observation: Observation, maximum_steps: int) -> TensorFrame:
    """Convert one public observation to reusable CPU tensors."""

    designated = observation["team_state"][0]["designated_possession_player"]
    if not 0 <= designated < 11:
        designated = observation["ball_owned_player"]
    opponent_designated = observation["team_state"][1][
        "designated_possession_player"
    ]
    if not 0 <= opponent_designated < 11:
        opponent_designated = (
            observation["ball_owned_player"]
            if observation["ball_owned_team"] == 1
            else 0
        )
    ball_position = observation["ball_position"]
    controlled_position = (
        observation["teams"][0][designated]["position"]
        if 0 <= designated < 11
        else ball_position
    )
    return TensorFrame(
        encode(observation, maximum_steps),
        torch.tensor(
            [
                player_features(player, ball_position, controlled_position)
                for player in observation["teams"][0]
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                player_features(player, ball_position, controlled_position)
                for player in observation["teams"][1]
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            context_features(observation, maximum_steps), dtype=torch.float32
        ),
        torch.tensor(designated if 0 <= designated < 11 else 0, dtype=torch.long),
        torch.tensor(
            opponent_designated if 0 <= opponent_designated < 11 else 0,
            dtype=torch.long,
        ),
    )


def batch_tensor_histories(
    histories: Sequence[Sequence[TensorFrame]],
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
    """Stack cached tensor frames into one device batch."""

    windows = [padded_history(history, history_length) for history in histories]
    batches = []
    for field in range(len(TensorFrame._fields)):
        batches.append(
            torch.stack(
                [torch.stack([frame[field] for frame in window]) for window in windows]
            ).to(device, non_blocking=True)
        )
    return tuple(batches)


class EntityTransformerBlock(nn.Module):
    """Apply self-attention and a residual feed-forward update to entity tokens."""

    def __init__(self, width: int, heads: int, feedforward_width: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Linear(feedforward_width, width),
        )

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class EntityTransformer(nn.Module):
    """Model interactions among both teams, the ball, and match context."""

    def __init__(
        self,
        width: int = ENTITY_WIDTH,
        heads: int = 8,
        layers: int = 3,
        feedforward_width: int = 512,
    ) -> None:
        super().__init__()
        self.output_width = width * 5
        self.player_encoder = nn.Sequential(
            nn.LayerNorm(PLAYER_FEATURES),
            nn.Linear(PLAYER_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.ball_encoder = nn.Sequential(
            nn.LayerNorm(BALL_FEATURES),
            nn.Linear(BALL_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.match_encoder = nn.Sequential(
            nn.LayerNorm(MATCH_FEATURES),
            nn.Linear(MATCH_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.type_embedding = nn.Embedding(4, width)
        self.slot_embedding = nn.Embedding(ENTITY_COUNT, width)
        self.own_anchor_embedding = nn.Parameter(torch.zeros(width))
        self.opponent_anchor_embedding = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(
            EntityTransformerBlock(width, heads, feedforward_width)
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.register_buffer(
            "type_indices",
            torch.tensor([0] * 11 + [1] * 11 + [2, 3], dtype=torch.long),
        )

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        weights = active.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        shape = own.shape[:-2]
        own_flat = own.reshape(-1, 11, PLAYER_FEATURES)
        opponent_flat = opponent.reshape(-1, 11, PLAYER_FEATURES)
        players = torch.cat((own_flat, opponent_flat), dim=1)
        context_flat = context.reshape(-1, CONTEXT_FEATURES)
        anchors_flat = anchor_indices.reshape(-1).clamp(0, 10)
        opponent_anchors_flat = opponent_anchor_indices.reshape(-1).clamp(0, 10)

        tokens = torch.cat(
            (
                self.player_encoder(players),
                self.ball_encoder(
                    context_flat[:, :BALL_FEATURES]
                ).unsqueeze(1),
                self.match_encoder(
                    context_flat[:, BALL_FEATURES:]
                ).unsqueeze(1),
            ),
            dim=1,
        )
        tokens = (
            tokens
            + self.type_embedding(self.type_indices).unsqueeze(0)
            + self.slot_embedding.weight.unsqueeze(0)
        )
        own_markers = nn.functional.one_hot(
            anchors_flat, num_classes=ENTITY_COUNT
        ).to(tokens.dtype)
        opponent_markers = nn.functional.one_hot(
            opponent_anchors_flat + 11, num_classes=ENTITY_COUNT
        ).to(tokens.dtype)
        tokens = (
            tokens
            + own_markers.unsqueeze(-1) * self.own_anchor_embedding
            + opponent_markers.unsqueeze(-1) * self.opponent_anchor_embedding
        )

        active_players = players[..., PLAYER_ACTIVE_INDEX] > 0.5
        context_active = torch.ones(
            active_players.shape[0], 2, dtype=torch.bool, device=players.device
        )
        padding_mask = ~torch.cat((active_players, context_active), dim=1)
        for block in self.blocks:
            tokens = block(tokens, padding_mask)
        tokens = self.output_norm(tokens)

        rows = torch.arange(tokens.shape[0], device=tokens.device)
        controlled = tokens[rows, anchors_flat]
        own_team = self._masked_mean(tokens[:, :11], active_players[:, :11])
        opponent_team = self._masked_mean(
            tokens[:, 11:22], active_players[:, 11:22]
        )
        output = torch.cat(
            (controlled, own_team, opponent_team, tokens[:, 22], tokens[:, 23]),
            dim=-1,
        )
        return output.reshape(*shape, self.output_width)
