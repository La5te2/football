"""Entity-Transformer actor-critic network for the Jackaroo policy."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .attention import EntityTransformer


POLICY_ARCHITECTURE = "entity-transformer"
POLICY_OBJECTIVE = "next-goal-wdl"
GLOBAL_WIDTH = 256
TEMPORAL_WIDTH = 384
HEAD_WIDTH = 256


class ActorCritic(nn.Module):
    """Encode public observation histories and predict actions and value."""

    def __init__(self, frame_size: int, action_count: int) -> None:
        super().__init__()
        self.entities = EntityTransformer()
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(frame_size),
            nn.Linear(frame_size, GLOBAL_WIDTH),
            nn.GELU(),
            nn.Linear(GLOBAL_WIDTH, GLOBAL_WIDTH),
            nn.GELU(),
        )
        frame_width = GLOBAL_WIDTH + self.entities.output_width
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(frame_width),
            nn.Linear(frame_width, TEMPORAL_WIDTH),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            TEMPORAL_WIDTH,
            TEMPORAL_WIDTH,
            num_layers=2,
            batch_first=True,
        )
        self.actor_encoder = nn.Sequential(
            nn.LayerNorm(TEMPORAL_WIDTH),
            nn.Linear(TEMPORAL_WIDTH, HEAD_WIDTH),
            nn.GELU(),
        )
        self.critic_encoder = nn.Sequential(
            nn.LayerNorm(TEMPORAL_WIDTH),
            nn.Linear(TEMPORAL_WIDTH, HEAD_WIDTH),
            nn.GELU(),
        )
        self.action_head = nn.Linear(HEAD_WIDTH, action_count)
        self.value_head = nn.Linear(HEAD_WIDTH, 1)

    def _encode_history(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        entities = self.entities(
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        sequence = self.frame_encoder(
            torch.cat((self.global_encoder(encoded), entities), dim=-1)
        )
        temporal, _ = self.temporal(sequence)
        return temporal[:, -1]

    def action_logits(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Actor output without evaluating the Critic head."""

        hidden = self._encode_history(
            encoded,
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        return self.action_head(self.actor_encoder(hidden))

    def forward(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_history(
            encoded,
            own,
            opponent,
            context,
            anchor_indices,
            opponent_anchor_indices,
        )
        return (
            self.action_head(self.actor_encoder(hidden)),
            self.value_head(self.critic_encoder(hidden)).squeeze(-1),
        )

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
