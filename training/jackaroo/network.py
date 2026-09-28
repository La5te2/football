"""Actor-critic network for the single-player action policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(256, action_count)
        self.value_head = nn.Linear(256, 1)

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observation)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)

    def distributions(
        self, observation: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        action_logits, value = self(observation)
        return Categorical(logits=action_logits), value
