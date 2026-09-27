"""Actor-critic network with independent player-recipient and action heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(
        self, observation_size: int, player_count: int, action_count: int
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.player_head = nn.Linear(256, player_count)
        self.action_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(observation)
        return (
            self.player_head(hidden),
            self.action_head(hidden),
            self.value_head(hidden).squeeze(-1),
        )

    def distributions(
        self, observation: torch.Tensor, player_mask: torch.Tensor
    ) -> tuple[Categorical, Categorical, torch.Tensor]:
        player_logits, action_logits, value = self(observation)
        player_logits = player_logits.masked_fill(~player_mask, -torch.inf)
        return Categorical(logits=player_logits), Categorical(
            logits=action_logits
        ), value
