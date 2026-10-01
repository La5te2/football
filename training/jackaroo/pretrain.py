"""TamakEri self-play collection and supervised Jackaroo initialization."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from .attention import (
    OPPONENT_POSSESSION_INDEX,
    OWN_POSSESSION_INDEX,
    PLAYER_ACTIVE_INDEX,
    TensorFrame,
    batch_tensor_frames,
    tensorize,
)
from .env import SelfPlayVectorFootballEnv
from .network import ActorCritic, RecurrentState
from .ppo import Rollout, Segment, Transition, compact_frame, pack_rollout
from .tamakeri import TamakEriTeacher


def _select_state(state: RecurrentState, indices: list[int]) -> RecurrentState:
    index = torch.tensor(indices, dtype=torch.long, device=state.entities.device)
    return RecurrentState(
        state.entities.index_select(0, index),
        state.global_state.index_select(0, index),
    )


def _put_state(
    state: RecurrentState, indices: list[int], selected: RecurrentState
) -> RecurrentState:
    index = torch.tensor(indices, dtype=torch.long, device=state.entities.device)
    entities = state.entities.clone()
    global_state = state.global_state.clone()
    entities.index_copy_(0, index, selected.entities)
    global_state.index_copy_(0, index, selected.global_state)
    return RecurrentState(entities, global_state)


def _clear_state(state: RecurrentState, index: int) -> RecurrentState:
    entities = state.entities.clone()
    global_state = state.global_state.clone()
    entities[index].zero_()
    global_state[index].zero_()
    return RecurrentState(entities, global_state)


def _batch_tensor(
    tensor: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    value = tensor.index_select(0, indices)
    if value.dtype == torch.float16:
        value = value.float()
    return value.to(device, non_blocking=True)


def collect_tamakeri_rollout(
    environment: SelfPlayVectorFootballEnv,
    teacher: TamakEriTeacher,
    policy: ActorCritic,
    games: int,
    sequence_length: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[Rollout, list[tuple[int, int]], list[tuple[int, int]]]:
    """Run complete TamakEri self-play matches and retain scored segments."""

    if games <= 0 or sequence_length <= 0:
        raise ValueError("pretraining games and sequence length must be positive")
    first_flight_size = min(environment.count, games)
    observations = environment.reset(first_flight_size)
    teacher.reset()
    device = next(policy.parameters()).device
    agent_count = 2 * environment.count
    recurrent = policy.initial_state(agent_count, device)
    pending: list[list[Transition]] = [[] for _ in range(agent_count)]
    retained: list[Segment] = []
    completed_segments: list[tuple[int, int]] = []
    completed_matches: list[tuple[int, int]] = []
    simulated_steps = flight = supervised_steps = 0
    report_interval = max(500 * environment.count, 1)
    next_report = report_interval
    policy.eval()

    while len(completed_matches) < games:
        flight += 1
        flight_size = min(environment.count, games - len(completed_matches))
        if flight > 1:
            reset_agents: list[int] = []
            for index in range(flight_size):
                observations[index] = environment.reset_one(index)
                reset_agents.extend((2 * index, 2 * index + 1))
            teacher.reset(reset_agents)
        active = [index < flight_size for index in range(environment.count)]
        if progress is not None:
            progress(
                f"flight={flight} start matches={len(completed_matches)}/{games} "
                f"active={flight_size}"
            )
        while any(active):
            agent_indices: list[int] = []
            frames: list[TensorFrame] = []
            teacher_observations: list[dict] = []
            for environment_index in range(environment.count):
                if not active[environment_index]:
                    continue
                for side in range(2):
                    observation = observations[environment_index][side]
                    if observation["is_in_play"]:
                        agent_indices.append(2 * environment_index + side)
                        frames.append(tensorize(observation))
                        teacher_observations.append(observation)

            decisions = [[[32] * 11, [32] * 11] for _ in range(environment.count)]
            step_data: dict[
                int, tuple[TensorFrame, RecurrentState | None, int, bool, float]
            ] = {}
            if frames:
                inputs = batch_tensor_frames(frames, device)
                before = _select_state(recurrent, agent_indices)
                with torch.inference_mode():
                    output = policy.step(*inputs, before)
                recurrent = _put_state(recurrent, agent_indices, output.state)
                teacher_decisions = teacher.decide(
                    teacher_observations, agent_indices
                )
                for local, (agent_index, teacher_decision) in enumerate(
                    zip(agent_indices, teacher_decisions)
                ):
                    environment_index, side = divmod(agent_index, 2)
                    decisions[environment_index][side] = teacher_decision.actions
                    supervised = teacher_decision.action is not None
                    action = teacher_decision.action if supervised else 0
                    supervised_steps += int(supervised)
                    sequence_start = len(pending[agent_index]) % sequence_length == 0
                    saved_state = (
                        RecurrentState(
                            before.entities[local:local + 1].detach().cpu().half(),
                            before.global_state[local:local + 1].detach().cpu().half(),
                        )
                        if sequence_start else None
                    )
                    step_data[agent_index] = (
                        frames[local], saved_state, action, supervised,
                        float(output.value[local]),
                    )

            next_observations, results, segment_done, infos = (
                environment.step_decisions(decisions, active)
            )
            simulated_steps += sum(active)
            for environment_index in range(environment.count):
                if not active[environment_index]:
                    continue
                for side in range(2):
                    agent_index = 2 * environment_index + side
                    if infos[environment_index]["action_applied"][side]:
                        frame, before, action, supervised, value = step_data[agent_index]
                        pending[agent_index].append(
                            Transition(
                                frame=compact_frame(frame),
                                next_frame=compact_frame(tensorize(
                                    next_observations[environment_index][side]
                                )),
                                state=before,
                                action=action,
                                old_log_probability=0.0,
                                old_value=value,
                                action_supervised=supervised,
                            )
                        )
                if segment_done[environment_index]:
                    result = results[environment_index]
                    completed_segments.append(
                        (int(result), infos[environment_index]["segment_steps"])
                    )
                    if result:
                        for side, side_result in ((0, result), (1, -result)):
                            agent_index = 2 * environment_index + side
                            if not pending[agent_index]:
                                raise RuntimeError(
                                    "a scored teacher segment has no decision"
                                )
                            pending[agent_index][-1].next_frame = compact_frame(
                                tensorize(next_observations[environment_index][side])
                            )
                            pending[agent_index][-1].terminal_class = (
                                1 if side_result == 1 else 2
                            )
                            retained.append(
                                Segment(pending[agent_index], int(side_result))
                            )
                    for side in range(2):
                        agent_index = 2 * environment_index + side
                        pending[agent_index] = []
                        recurrent = _clear_state(recurrent, agent_index)
                    if infos[environment_index]["match_done"]:
                        completed_matches.append(
                            tuple(infos[environment_index]["goals"])
                        )
                        active[environment_index] = False
            observations = next_observations
            if progress is not None and simulated_steps >= next_report:
                progress(
                    f"flight={flight} matches={len(completed_matches)}/{games} "
                    f"engine_steps={simulated_steps} active={sum(active)}/{flight_size} "
                    f"segments={len(completed_segments)} "
                    f"scored={sum(result != 0 for result, _ in completed_segments)} "
                    f"labels={supervised_steps}"
                )
                next_report += report_interval
        if progress is not None:
            progress(
                f"flight={flight} complete matches={len(completed_matches)}/{games} "
                f"segments={len(completed_segments)} "
                f"scored={sum(result != 0 for result, _ in completed_segments)} "
                f"labels={supervised_steps}"
            )

    return (
        pack_rollout(retained, sequence_length),
        completed_segments,
        completed_matches,
    )


def pretrain(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 2,
    batch_size: int = 256,
    value_coefficient: float = 0.5,
    transition_coefficient: float = 0.1,
    action_value_coefficient: float = 0.1,
    control_coefficient: float = 0.05,
    space_coefficient: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Initialize Jackaroo from teacher actions and scored segment outcomes."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("pretraining epochs and batch size must be positive")
    sequence_count, sequence_length = rollout.encoded.shape[:2]
    sequences_per_batch = max(batch_size // sequence_length, 1)
    batch_count = math.ceil(sequence_count / sequences_per_batch)
    report_interval = max(batch_count // 4, 1)
    names = (
        "behavior", "value", "outcome", "terminal", "latent",
        "action_value", "control", "space", "accuracy",
    )
    totals = {name: 0.0 for name in names}
    weights = {name: 0 for name in names}
    device = next(policy.parameters()).device
    policy.train()

    for epoch in range(1, epochs + 1):
        epoch_totals = {name: 0.0 for name in names}
        epoch_weights = {name: 0 for name in names}
        batches = torch.randperm(sequence_count).split(sequences_per_batch)
        for batch_number, indices in enumerate(batches, start=1):
            encoded = _batch_tensor(rollout.encoded, indices, device)
            own = _batch_tensor(rollout.own, indices, device)
            opponent = _batch_tensor(rollout.opponent, indices, device)
            context = _batch_tensor(rollout.context, indices, device)
            anchor_indices = _batch_tensor(rollout.anchor_indices, indices, device)
            opponent_anchor_indices = _batch_tensor(
                rollout.opponent_anchor_indices, indices, device
            )
            initial_entity_state = _batch_tensor(
                rollout.initial_entity_state, indices, device
            )
            initial_global_state = _batch_tensor(
                rollout.initial_global_state, indices, device
            )
            valid = _batch_tensor(rollout.valid, indices, device)
            action_supervision = _batch_tensor(
                rollout.action_supervision, indices, device
            )
            actions = _batch_tensor(rollout.actions, indices, device)
            returns = _batch_tensor(rollout.returns, indices, device)
            terminal_classes = _batch_tensor(
                rollout.terminal_classes, indices, device
            )
            next_own = _batch_tensor(rollout.next_own, indices, device)
            next_opponent = _batch_tensor(rollout.next_opponent, indices, device)
            next_context = _batch_tensor(rollout.next_context, indices, device)
            next_anchor_indices = _batch_tensor(
                rollout.next_anchor_indices, indices, device
            )
            supervised = valid & action_supervision
            value_supervised = valid
            terminal_supervised = supervised
            action_value_supervised = supervised
            output = policy.sequence(
                encoded, own, opponent, context,
                anchor_indices, opponent_anchor_indices,
                RecurrentState(initial_entity_state, initial_global_state),
                valid,
            )
            if supervised.any():
                behavior_loss = nn.functional.cross_entropy(
                    output.logits[supervised], actions[supervised]
                )
                accuracy = (
                    output.logits[supervised].argmax(dim=-1)
                    == actions[supervised]
                ).float().mean()
            else:
                behavior_loss = output.logits.sum() * 0.0
                accuracy = output.logits.sum() * 0.0
            if value_supervised.any():
                value_loss = nn.functional.mse_loss(
                    output.value[value_supervised], returns[value_supervised]
                )
            else:
                value_loss = output.value.sum() * 0.0

            action_index = actions.unsqueeze(-1)
            selected_outcome = output.predicted_outcome.gather(
                2,
                action_index[..., None].expand(
                    -1, -1, 1, output.predicted_outcome.shape[-1]
                ),
            ).squeeze(2)
            selected_terminal = output.terminal_logits.gather(
                2, action_index[..., None].expand(-1, -1, 1, 3)
            ).squeeze(2)
            selected_latent = output.predicted_latent.gather(
                2,
                action_index[..., None].expand(
                    -1, -1, 1, output.predicted_latent.shape[-1]
                ),
            ).squeeze(2)
            selected_action_value = output.action_values.gather(
                2, action_index
            ).squeeze(-1)
            target_outcome = policy.observed_outcome(
                next_own.reshape(-1, 11, next_own.shape[-1]),
                next_opponent.reshape(-1, 11, next_opponent.shape[-1]),
                next_context.reshape(-1, next_context.shape[-1]),
                next_anchor_indices.reshape(-1),
            ).reshape(*valid.shape, -1)
            if supervised.any():
                outcome_loss = nn.functional.mse_loss(
                    selected_outcome[supervised], target_outcome[supervised]
                )
                if terminal_supervised.any():
                    terminal_loss = nn.functional.cross_entropy(
                        selected_terminal[terminal_supervised],
                        terminal_classes[terminal_supervised],
                    )
                else:
                    terminal_loss = selected_terminal.sum() * 0.0
                if action_value_supervised.any():
                    action_value_loss = nn.functional.mse_loss(
                        selected_action_value[action_value_supervised],
                        returns[action_value_supervised],
                    )
                else:
                    action_value_loss = selected_action_value.sum() * 0.0
            else:
                outcome_loss = selected_outcome.sum() * 0.0
                terminal_loss = selected_terminal.sum() * 0.0
                action_value_loss = selected_action_value.sum() * 0.0
            latent_valid = supervised[:, :-1] & valid[:, 1:]
            latent_valid &= terminal_classes[:, :-1] == 0
            if latent_valid.any():
                latent_loss = nn.functional.smooth_l1_loss(
                    selected_latent[:, :-1][latent_valid],
                    output.latent[:, 1:].detach()[latent_valid],
                )
            else:
                latent_loss = selected_latent.sum() * 0.0

            players = torch.cat((own, opponent), dim=2)
            reach_target = players[..., 20]
            reach_valid = valid[..., None] & (
                players[..., PLAYER_ACTIVE_INDEX] > 0.5
            )
            reach_valid &= reach_target > 0.0
            if reach_valid.any():
                control_loss = nn.functional.smooth_l1_loss(
                    torch.log1p(output.reach_ball[reach_valid] * 10.0),
                    torch.log1p(reach_target[reach_valid] * 10.0),
                )
            else:
                control_loss = output.reach_ball.sum() * 0.0

            space_target = policy.control.analytic_space_value_target(
                own.reshape(-1, 11, own.shape[-1]),
                opponent.reshape(-1, 11, opponent.shape[-1]),
            ).reshape(*valid.shape, 2, -1)
            possession = torch.stack(
                (
                    context[..., OWN_POSSESSION_INDEX] > 0.5,
                    context[..., OPPONENT_POSSESSION_INDEX] > 0.5,
                ),
                dim=-1,
            )
            space_valid = (valid[..., None] & possession)[..., None].expand_as(
                output.space_values
            )
            if space_valid.any():
                space_loss = nn.functional.smooth_l1_loss(
                    output.space_values[space_valid], space_target[space_valid]
                )
            else:
                space_loss = output.space_values.sum() * 0.0

            loss = (
                behavior_loss
                + value_coefficient * value_loss
                + transition_coefficient * (
                    outcome_loss + terminal_loss + latent_loss
                )
                + action_value_coefficient * action_value_loss
                + control_coefficient * control_loss
                + space_coefficient * space_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("pretraining loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("pretraining gradient norm is NaN or Inf")
            optimizer.step()

            supervised_count = int(supervised.sum())
            metric_weights = {
                "behavior": supervised_count,
                "value": int(value_supervised.sum()),
                "outcome": supervised_count,
                "terminal": int(terminal_supervised.sum()),
                "latent": int(latent_valid.sum()),
                "action_value": int(action_value_supervised.sum()),
                "control": int(reach_valid.sum()),
                "space": int(space_valid.sum()),
                "accuracy": supervised_count,
            }
            metrics = {
                "behavior": behavior_loss.item(), "value": value_loss.item(),
                "outcome": outcome_loss.item(), "terminal": terminal_loss.item(),
                "latent": latent_loss.item(),
                "action_value": action_value_loss.item(),
                "control": control_loss.item(), "space": space_loss.item(),
                "accuracy": accuracy.item(),
            }
            for name, metric in metrics.items():
                weight = metric_weights[name]
                totals[name] += metric * weight
                weights[name] += weight
                epoch_totals[name] += metric * weight
                epoch_weights[name] += weight
            if progress is not None and (
                batch_number % report_interval == 0 or batch_number == batch_count
            ):
                progress(f"epoch={epoch}/{epochs} batch={batch_number}/{batch_count}")
        if progress is not None:
            progress(
                f"epoch={epoch}/{epochs} complete "
                f"behavior={epoch_totals['behavior'] / max(epoch_weights['behavior'], 1):.4f} "
                f"value={epoch_totals['value'] / max(epoch_weights['value'], 1):.4f} "
                f"accuracy={epoch_totals['accuracy'] / max(epoch_weights['accuracy'], 1):.4f}"
            )
    return {
        name: value / max(weights[name], 1) for name, value in totals.items()
    }
