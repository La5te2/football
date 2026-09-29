"""Recurrent actor-critic network for the Jackaroo policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .attention import RelationEncoder


class ActorCritic(nn.Module):
    """Encode public observation histories and predict actions and value."""

    def __init__(self, frame_size: int, action_count: int) -> None:
        super().__init__()
        self.relation = RelationEncoder()
        self.global_encoder = nn.Sequential(
            nn.Linear(frame_size, 128),
            nn.Tanh(),
        )
        self.temporal = nn.GRU(
            128 + self.relation.output_width, 256, batch_first=True
        )
        self.action_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation, _, _ = self.relation(
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        sequence = torch.cat((self.global_encoder(encoded), relation), dim=-1)
        temporal, _ = self.temporal(sequence)
        hidden = temporal[:, -1]
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)

    def distributions(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        action_logits, value = self(
            encoded,
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        return Categorical(logits=action_logits), value
