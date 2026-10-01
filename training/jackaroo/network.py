"""Persistent entity-control actor-critic network for Jackaroo."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Categorical

from .attention import (
    ENTITY_COUNT,
    ENTITY_WIDTH,
    PLAYER_ACTIVE_INDEX,
    EntityTransformer,
)


POLICY_ARCHITECTURE = "persistent-entity-control-transition-transformer"
POLICY_OBJECTIVE = "next-goal-wdl"
ENTITY_STATE_WIDTH = 64
GLOBAL_STATE_WIDTH = 256
RELATION_WIDTH = 256
ACTION_WIDTH = 64
CONTROL_WIDTH = 128
CONTROL_GRID_X = 12
CONTROL_GRID_Y = 8
CONTROL_LOCATIONS = CONTROL_GRID_X * CONTROL_GRID_Y
ACTION_FEATURES = 18
OBSERVED_OUTCOME_WIDTH = 6 + 22 * 4 + CONTROL_LOCATIONS + 3 + 12 + 11


class RecurrentState(NamedTuple):
    """Persistent states attached to entities and the complete match view."""

    entities: torch.Tensor
    global_state: torch.Tensor


class PolicyOutput(NamedTuple):
    """One recurrent policy evaluation and its auxiliary predictions."""

    logits: torch.Tensor
    value: torch.Tensor
    state: RecurrentState
    latent: torch.Tensor
    predicted_latent: torch.Tensor
    predicted_outcome: torch.Tensor
    terminal_logits: torch.Tensor
    action_values: torch.Tensor
    reach_ball: torch.Tensor


class SequenceOutput(NamedTuple):
    """Policy outputs for a batch of contiguous causal sequences."""

    logits: torch.Tensor
    value: torch.Tensor
    state: RecurrentState
    latent: torch.Tensor
    predicted_latent: torch.Tensor
    predicted_outcome: torch.Tensor
    terminal_logits: torch.Tensor
    action_values: torch.Tensor
    reach_ball: torch.Tensor


def _action_feature_table() -> torch.Tensor:
    """Describe direction, football function, and press/release phase."""

    table = torch.zeros(32, ACTION_FEATURES)
    diagonal = math.sqrt(0.5)
    directions = (
        (-1.0, 0.0),
        (-diagonal, diagonal),
        (0.0, 1.0),
        (diagonal, diagonal),
        (1.0, 0.0),
        (diagonal, -diagonal),
        (0.0, -1.0),
        (-diagonal, -diagonal),
    )
    for action, direction in enumerate(directions, start=1):
        table[action, :2] = torch.tensor(direction)

    # Families are idle, direction, and the eleven football buttons 9..19.
    table[0, 2] = 1.0
    table[1:9, 3] = 1.0
    for action in range(9, 20):
        table[action, 3 + action - 8] = 1.0
    table[20, 3] = 1.0
    for action in range(21, 32):
        table[action, 3 + action - 20] = 1.0

    table[0, 15] = 1.0
    table[1:20, 16] = 1.0
    table[20:32, 17] = 1.0
    return table


class ControlField(nn.Module):
    """Compute a DSS prior and an engine-calibrated spatial control field."""

    def __init__(self) -> None:
        super().__init__()
        xs = torch.linspace(-1.0, 1.0, CONTROL_GRID_X)
        ys = torch.linspace(-0.42, 0.42, CONTROL_GRID_Y)
        self.register_buffer("locations", torch.cartesian_prod(xs, ys))
        self.register_buffer("maximum_speed", torch.tensor(0.012))
        self.entity_context = nn.Linear(ENTITY_WIDTH + ENTITY_STATE_WIDTH, 16)
        self.reach_correction = nn.Sequential(
            nn.Linear(24, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        self.position_encoder = nn.Sequential(
            nn.LayerNorm(136),
            nn.Linear(136, CONTROL_WIDTH),
            nn.GELU(),
            nn.Linear(CONTROL_WIDTH, CONTROL_WIDTH),
        )

    def _reach(
        self,
        players: torch.Tensor,
        entity_context: torch.Tensor,
        locations: torch.Tensor,
    ) -> torch.Tensor:
        positions = players[..., :2]
        velocities = players[..., 3:5]
        difference = locations[:, None] - positions[:, :, None]
        distance = difference.square().sum(dim=-1).sqrt().clamp_min(1e-5)
        speed = velocities.square().sum(dim=-1).sqrt()
        cosine = (
            velocities[:, :, None] * difference
        ).sum(dim=-1) / (speed[:, :, None] * distance).clamp_min(1e-5)
        tired = players[..., 13].clamp(0.0, 1.0)
        maximum_speed = self.maximum_speed * (1.0 - 0.35 * tired)
        effective_speed = (
            speed[:, :, None] * cosine + maximum_speed[:, :, None]
        ).mul(0.5).clamp_min(5e-4)
        base_time = distance / effective_speed * 0.001
        scalar = torch.stack(
            (
                difference[..., 0],
                difference[..., 1],
                distance,
                cosine,
                speed[:, :, None].expand_as(distance),
                tired[:, :, None].expand_as(distance),
                players[..., 17][:, :, None].expand_as(distance),
                players[..., 20][:, :, None].expand_as(distance),
            ),
            dim=-1,
        )
        token_context = self.entity_context(entity_context)[:, :, None]
        token_context = token_context.expand(-1, -1, locations.shape[1], -1)
        correction = self.reach_correction(
            torch.cat((scalar, token_context), dim=-1)
        ).squeeze(-1)
        return base_time * torch.exp(1.5 * torch.tanh(correction))

    @staticmethod
    def _control_from_times(
        times: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        times = torch.where(
            active[:, :, None], times, torch.full_like(times, 1e4)
        )
        own_times = times[:, :11]
        opponent_times = times[:, 11:]
        own_time = -0.1 * torch.logsumexp(-own_times / 0.1, dim=1)
        opponent_time = -0.1 * torch.logsumexp(-opponent_times / 0.1, dim=1)
        margin = opponent_time - own_time
        return torch.sigmoid(margin / 0.15), own_times, opponent_times

    def analytic_control(
        self, own: torch.Tensor, opponent: torch.Tensor
    ) -> torch.Tensor:
        """Return a parameter-free DSS target for an observed public state."""

        players = torch.cat((own, opponent), dim=1)
        positions = players[..., :2]
        velocities = players[..., 3:5]
        locations = self.locations[None].expand(players.shape[0], -1, -1)
        difference = locations[:, None] - positions[:, :, None]
        distance = difference.square().sum(dim=-1).sqrt().clamp_min(1e-5)
        speed = velocities.square().sum(dim=-1).sqrt()
        cosine = (
            velocities[:, :, None] * difference
        ).sum(dim=-1) / (speed[:, :, None] * distance).clamp_min(1e-5)
        maximum_speed = self.maximum_speed * (
            1.0 - 0.35 * players[..., 13].clamp(0.0, 1.0)
        )
        effective_speed = (
            speed[:, :, None] * cosine + maximum_speed[:, :, None]
        ).mul(0.5).clamp_min(5e-4)
        times = distance / effective_speed * 0.001
        active = players[..., PLAYER_ACTIVE_INDEX] > 0.5
        control, _, _ = self._control_from_times(times, active)
        return control

    def forward(
        self,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        entity_tokens: torch.Tensor,
        entity_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        players = torch.cat((own, opponent), dim=1)
        active = players[..., PLAYER_ACTIVE_INDEX] > 0.5
        locations = self.locations[None].expand(players.shape[0], -1, -1)
        reach_context = torch.cat(
            (entity_tokens[:, :22], entity_state[:, :22]), dim=-1
        )
        times = self._reach(players, reach_context, locations)
        control, own_times, opponent_times = self._control_from_times(times, active)
        own_weights = torch.softmax(-own_times / 0.1, dim=1)
        opponent_weights = torch.softmax(-opponent_times / 0.1, dim=1)
        own_state = torch.einsum("bik,bid->bkd", own_weights, entity_state[:, :11])
        opponent_state = torch.einsum(
            "bik,bid->bkd", opponent_weights, entity_state[:, 11:22]
        )
        ball_position = context[:, :2, None].transpose(1, 2)
        ball_velocity = context[:, 3:5, None].transpose(1, 2)
        ball_velocity = ball_velocity.expand(-1, CONTROL_LOCATIONS, -1)
        margin = opponent_times.min(dim=1).values - own_times.min(dim=1).values
        position_input = torch.cat(
            (
                locations,
                locations - ball_position,
                ball_velocity,
                control[:, :, None],
                torch.tanh(margin / 0.2)[:, :, None],
                own_state,
                opponent_state,
            ),
            dim=-1,
        )
        ball_locations = context[:, :2, None].transpose(1, 2)
        reach_ball = self._reach(players, reach_context, ball_locations).squeeze(-1)
        return self.position_encoder(position_input), control, reach_ball


class ActorCritic(nn.Module):
    """Predict actions and the single next-goal EPV from persistent state."""

    def __init__(self, frame_size: int, action_count: int) -> None:
        super().__init__()
        if action_count != 32:
            raise ValueError("Jackaroo requires the engine's 32 atomic actions")
        self.frame_size = frame_size
        self.action_count = action_count
        self.entities = EntityTransformer()
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(frame_size),
            nn.Linear(frame_size, GLOBAL_STATE_WIDTH),
            nn.GELU(),
            nn.Linear(GLOBAL_STATE_WIDTH, GLOBAL_STATE_WIDTH),
        )
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(self.entities.output_width),
            nn.Linear(self.entities.output_width, GLOBAL_STATE_WIDTH),
            nn.GELU(),
        )
        self.entity_input = nn.Linear(ENTITY_WIDTH, ENTITY_STATE_WIDTH)
        self.entity_temporal = nn.GRUCell(ENTITY_STATE_WIDTH, ENTITY_STATE_WIDTH)
        self.control = ControlField()
        self.control_global = nn.Linear(CONTROL_WIDTH, GLOBAL_STATE_WIDTH)
        self.global_temporal = nn.GRUCell(GLOBAL_STATE_WIDTH, GLOBAL_STATE_WIDTH)
        self.relation_state = nn.Sequential(
            nn.LayerNorm(
                GLOBAL_STATE_WIDTH + ENTITY_WIDTH + ENTITY_STATE_WIDTH + CONTROL_WIDTH
            ),
            nn.Linear(
                GLOBAL_STATE_WIDTH + ENTITY_WIDTH + ENTITY_STATE_WIDTH + CONTROL_WIDTH,
                RELATION_WIDTH,
            ),
            nn.GELU(),
        )
        self.register_buffer("action_features", _action_feature_table())
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURES, ACTION_WIDTH),
            nn.GELU(),
            nn.Linear(ACTION_WIDTH, ACTION_WIDTH),
        )
        transition_input = GLOBAL_STATE_WIDTH + ENTITY_STATE_WIDTH + ACTION_WIDTH
        transition_output = GLOBAL_STATE_WIDTH + OBSERVED_OUTCOME_WIDTH + 3
        self.transition_head = nn.Sequential(
            nn.LayerNorm(transition_input),
            nn.Linear(transition_input, 512),
            nn.GELU(),
            nn.Linear(512, transition_output),
        )
        self.outcome_encoder = nn.Sequential(
            nn.LayerNorm(OBSERVED_OUTCOME_WIDTH),
            nn.Linear(OBSERVED_OUTCOME_WIDTH, CONTROL_WIDTH),
            nn.GELU(),
        )
        self.epv = nn.Sequential(
            nn.LayerNorm(GLOBAL_STATE_WIDTH),
            nn.Linear(GLOBAL_STATE_WIDTH, RELATION_WIDTH),
            nn.GELU(),
            nn.Linear(RELATION_WIDTH, 1),
            nn.Tanh(),
        )
        actor_input = RELATION_WIDTH + ACTION_WIDTH + CONTROL_WIDTH
        self.actor = nn.Sequential(
            nn.LayerNorm(actor_input),
            nn.Linear(actor_input, RELATION_WIDTH),
            nn.GELU(),
            nn.Linear(RELATION_WIDTH, 1),
        )
        self.action_value_gate = nn.Parameter(torch.zeros(()))

    def initial_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        return RecurrentState(
            torch.zeros(batch_size, ENTITY_COUNT, ENTITY_STATE_WIDTH, device=device),
            torch.zeros(batch_size, GLOBAL_STATE_WIDTH, device=device),
        )

    def observed_outcome(
        self,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Build the public next-state target used by the transition model."""

        players = torch.cat((own, opponent), dim=1)
        player_motion = players[..., (0, 1, 3, 4)].reshape(players.shape[0], -1)
        control = self.control.analytic_control(own, opponent)
        anchors = nn.functional.one_hot(
            anchor_indices.clamp(0, 10), num_classes=11
        ).to(context.dtype)
        return torch.cat(
            (
                context[:, :6],
                player_motion,
                control,
                context[:, 31:34],
                context[:, 34:46],
                anchors,
            ),
            dim=-1,
        )

    def step(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
        state: RecurrentState,
    ) -> PolicyOutput:
        entity_tokens, entity_summary = self.entities(
            own, opponent, context, anchor_indices, opponent_anchor_indices
        )
        entity_input = self.entity_input(entity_tokens)
        entity_state = self.entity_temporal(
            entity_input.reshape(-1, ENTITY_STATE_WIDTH),
            state.entities.reshape(-1, ENTITY_STATE_WIDTH),
        ).reshape(-1, ENTITY_COUNT, ENTITY_STATE_WIDTH)
        players = torch.cat((own, opponent), dim=1)
        active_players = players[..., PLAYER_ACTIVE_INDEX] > 0.5
        active = torch.cat(
            (
                active_players,
                torch.ones(
                    active_players.shape[0], 2, dtype=torch.bool,
                    device=active_players.device,
                ),
            ),
            dim=1,
        )
        entity_state = torch.where(
            active[:, :, None], entity_state, torch.zeros_like(entity_state)
        )
        control_tokens, _, reach_ball = self.control(
            own, opponent, context, entity_tokens, entity_state
        )
        control_summary = control_tokens.mean(dim=1)
        current_global = (
            self.global_encoder(encoded)
            + self.relation_encoder(entity_summary)
            + self.control_global(control_summary)
        )
        global_state = self.global_temporal(current_global, state.global_state)
        rows = torch.arange(encoded.shape[0], device=encoded.device)
        anchors = anchor_indices.clamp(0, 10)
        controlled_token = entity_tokens[rows, anchors]
        controlled_state = entity_state[rows, anchors]
        relation = self.relation_state(
            torch.cat(
                (global_state, controlled_token, controlled_state, control_summary),
                dim=-1,
            )
        )

        action = self.action_encoder(self.action_features)
        action = action[None].expand(encoded.shape[0], -1, -1)
        transition_context = torch.cat(
            (global_state, controlled_state), dim=-1
        )[:, None].expand(-1, self.action_count, -1)
        transition = self.transition_head(
            torch.cat((transition_context, action), dim=-1)
        )
        latent_delta, predicted_outcome, terminal_logits = torch.split(
            transition, (GLOBAL_STATE_WIDTH, OBSERVED_OUTCOME_WIDTH, 3), dim=-1
        )
        predicted_latent = global_state[:, None] + torch.tanh(latent_delta)
        next_epv = self.epv(predicted_latent).squeeze(-1)
        terminal_probability = torch.softmax(terminal_logits, dim=-1)
        action_values = (
            terminal_probability[..., 1]
            - terminal_probability[..., 2]
            + terminal_probability[..., 0] * next_epv
        )
        relation_expanded = relation[:, None].expand(-1, self.action_count, -1)
        outcome_features = self.outcome_encoder(predicted_outcome.detach())
        base_logits = self.actor(
            torch.cat((relation_expanded, action, outcome_features), dim=-1)
        ).squeeze(-1)
        logits = base_logits + self.action_value_gate * action_values.detach()
        value = self.epv(global_state).squeeze(-1)
        return PolicyOutput(
            logits,
            value,
            RecurrentState(entity_state, global_state),
            global_state,
            predicted_latent,
            predicted_outcome,
            terminal_logits,
            action_values,
            reach_ball,
        )

    def sequence(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
        initial_state: RecurrentState,
        valid: torch.Tensor,
    ) -> SequenceOutput:
        """Evaluate padded, contiguous sequences with truncated recurrence."""

        state = initial_state
        outputs: list[PolicyOutput] = []
        for index in range(encoded.shape[1]):
            output = self.step(
                encoded[:, index], own[:, index], opponent[:, index],
                context[:, index], anchor_indices[:, index],
                opponent_anchor_indices[:, index], state,
            )
            keep = valid[:, index].to(encoded.dtype)
            state = RecurrentState(
                output.state.entities * keep[:, None, None]
                + state.entities * (1.0 - keep[:, None, None]),
                output.state.global_state * keep[:, None]
                + state.global_state * (1.0 - keep[:, None]),
            )
            outputs.append(output)
        return SequenceOutput(
            torch.stack([output.logits for output in outputs], dim=1),
            torch.stack([output.value for output in outputs], dim=1),
            state,
            torch.stack([output.latent for output in outputs], dim=1),
            torch.stack([output.predicted_latent for output in outputs], dim=1),
            torch.stack([output.predicted_outcome for output in outputs], dim=1),
            torch.stack([output.terminal_logits for output in outputs], dim=1),
            torch.stack([output.action_values for output in outputs], dim=1),
            torch.stack([output.reach_ball for output in outputs], dim=1),
        )

    def distributions(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[Categorical, PolicyOutput]:
        output = self.step(
            encoded, own, opponent, context, anchor_indices,
            opponent_anchor_indices, state,
        )
        return Categorical(logits=output.logits), output

    def forward(
        self,
        encoded: torch.Tensor,
        own: torch.Tensor,
        opponent: torch.Tensor,
        context: torch.Tensor,
        anchor_indices: torch.Tensor,
        opponent_anchor_indices: torch.Tensor,
        entity_state: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """TorchScript-friendly persistent inference entry point."""

        output = self.step(
            encoded, own, opponent, context, anchor_indices,
            opponent_anchor_indices, RecurrentState(entity_state, global_state),
        )
        return (
            output.logits,
            output.value,
            output.state.entities,
            output.state.global_state,
        )
