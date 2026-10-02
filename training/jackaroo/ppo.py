"""Recurrent PPO over complete next-goal football segments."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical

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


@dataclass
class Transition:
    frame: TensorFrame
    next_frame: TensorFrame
    state: RecurrentState | None
    action: int
    old_log_probability: float
    old_value: float
    terminal_class: int = 0
    action_supervised: bool = True


@dataclass
class Segment:
    transitions: list[Transition]
    result: int


@dataclass
class Rollout:
    """Padded contiguous sequences and their segment-level WDL targets."""

    encoded: torch.Tensor
    own: torch.Tensor
    opponent: torch.Tensor
    context: torch.Tensor
    anchor_indices: torch.Tensor
    opponent_anchor_indices: torch.Tensor
    next_own: torch.Tensor
    next_opponent: torch.Tensor
    next_context: torch.Tensor
    next_anchor_indices: torch.Tensor
    initial_entity_state: torch.Tensor
    initial_global_state: torch.Tensor
    valid: torch.Tensor
    policy_valid: torch.Tensor
    terminal_supervision: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminal_classes: torch.Tensor
    action_supervision: torch.Tensor
    reward_total: float
    transition_count: int


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


def _stack_padded_frames(
    chunks: list[list[Transition]],
    length: int,
    next_frame: bool,
) -> tuple[torch.Tensor, ...]:
    fields: list[torch.Tensor] = []
    for field in range(len(TensorFrame._fields)):
        sequences = []
        for chunk in chunks:
            values = [
                (transition.next_frame if next_frame else transition.frame)[field]
                for transition in chunk
            ]
            value = torch.stack(values)
            padding = (length - len(values), *value.shape[1:])
            if padding[0]:
                value = torch.cat((value, torch.zeros(padding, dtype=value.dtype)))
            sequences.append(value)
        stacked = torch.stack(sequences)
        if stacked.is_floating_point():
            stacked = stacked.to(torch.float16)
        fields.append(stacked)
    return tuple(fields)


def compact_frame(frame: TensorFrame) -> TensorFrame:
    """Store a rollout frame on the CPU without retaining full-precision buffers."""

    return TensorFrame(*(
        value.to(dtype=torch.float16, device="cpu")
        if value.is_floating_point() else value.to(device="cpu")
        for value in frame
    ))


def _batch_tensor(
    tensor: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Load one CPU rollout slice onto the policy device."""

    value = tensor.index_select(0, indices)
    if value.dtype == torch.float16:
        value = value.float()
    return value.to(device, non_blocking=True)


def pack_rollout(segments: list[Segment], sequence_length: int) -> Rollout:
    chunks: list[list[Transition]] = []
    chunk_results: list[int] = []
    chunk_ends_segment: list[bool] = []
    for segment in segments:
        for start in range(0, len(segment.transitions), sequence_length):
            chunks.append(segment.transitions[start:start + sequence_length])
            chunk_results.append(segment.result)
            chunk_ends_segment.append(
                start + sequence_length >= len(segment.transitions)
            )
    if not chunks:
        raise RuntimeError("rollout contains no retained segment")

    current = _stack_padded_frames(chunks, sequence_length, False)
    following = _stack_padded_frames(chunks, sequence_length, True)
    sequence_count = len(chunks)
    valid = torch.zeros(sequence_count, sequence_length, dtype=torch.bool)
    policy_valid = torch.zeros(sequence_count, sequence_length, dtype=torch.bool)
    terminal_supervision = torch.zeros(
        sequence_count, sequence_length, dtype=torch.bool
    )
    actions = torch.zeros(sequence_count, sequence_length, dtype=torch.long)
    old_log_probabilities = torch.zeros(sequence_count, sequence_length)
    old_values = torch.zeros(sequence_count, sequence_length)
    returns = torch.zeros(sequence_count, sequence_length)
    terminal_classes = torch.zeros(sequence_count, sequence_length, dtype=torch.long)
    action_supervision = torch.zeros(sequence_count, sequence_length, dtype=torch.bool)
    initial_entities = []
    initial_globals = []
    transition_count = 0
    for index, (chunk, result, ends_segment) in enumerate(
        zip(chunks, chunk_results, chunk_ends_segment)
    ):
        count = len(chunk)
        transition_count += count
        valid[index, :count] = True
        if result:
            policy_valid[index, :count] = True
        terminal_supervision[index, :count] = True
        if result == 0 and ends_segment:
            terminal_supervision[index, count - 1] = False
        actions[index, :count] = torch.tensor([item.action for item in chunk])
        old_log_probabilities[index, :count] = torch.tensor(
            [item.old_log_probability for item in chunk]
        )
        old_values[index, :count] = torch.tensor([item.old_value for item in chunk])
        returns[index, :count] = float(result)
        terminal_classes[index, :count] = torch.tensor(
            [item.terminal_class for item in chunk]
        )
        action_supervision[index, :count] = torch.tensor(
            [item.action_supervised for item in chunk]
        )
        if chunk[0].state is None:
            raise RuntimeError("sequence start is missing its recurrent state")
        initial_entities.append(chunk[0].state.entities)
        initial_globals.append(chunk[0].state.global_state)

    return Rollout(
        *current,
        following[1],
        following[2],
        following[3],
        following[4],
        torch.cat(initial_entities).to(torch.float16),
        torch.cat(initial_globals).to(torch.float16),
        valid,
        policy_valid,
        terminal_supervision,
        actions,
        old_log_probabilities,
        returns - old_values,
        returns,
        terminal_classes,
        action_supervision,
        float(sum(segment.result for segment in segments)),
        transition_count,
    )


def collect_self_play_rollout(
    environment: SelfPlayVectorFootballEnv,
    policy: ActorCritic,
    games: int,
    sequence_length: int = 32,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    Rollout,
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Collect full matches with independently scheduled native simulations."""

    if games <= 0 or sequence_length <= 0:
        raise ValueError("rollout games and sequence length must be positive")
    initial_count = min(environment.count, games)
    observations = environment.reset(initial_count)
    device = next(policy.parameters()).device
    agent_count = 2 * environment.count
    recurrent = policy.initial_state(agent_count, device)
    pending: list[list[Transition]] = [[] for _ in range(agent_count)]
    retained: list[Segment] = []
    completed_segments: list[tuple[int, int]] = []
    completed_matches: list[tuple[int, int]] = []
    simulated_steps = 0
    report_interval = max(500 * environment.count, 1)
    next_report = report_interval
    policy.eval()
    ready = set(range(initial_count))
    reset_slots: set[int] = set()
    started_matches = initial_count
    StepData = dict[
        int, tuple[TensorFrame, RecurrentState | None, int, float, float]
    ]
    inflight: dict[Future, tuple[int, StepData]] = {}

    def submit_ready(executor: ThreadPoolExecutor) -> None:
        nonlocal recurrent
        if not ready:
            return
        slots = sorted(ready)
        ready.clear()
        decision_indices: list[int] = []
        frames: list[TensorFrame] = []
        for environment_index in slots:
            for side in range(2):
                observation = observations[environment_index][side]
                if observation["is_in_play"]:
                    decision_indices.append(2 * environment_index + side)
                    frames.append(tensorize(observation))
        decisions = {
            index: [[32] * 11, [32] * 11] for index in slots
        }
        step_data: dict[int, StepData] = {index: {} for index in slots}
        if frames:
            inputs = batch_tensor_frames(frames, device)
            before = _select_state(recurrent, decision_indices)
            with torch.inference_mode():
                distribution, output = policy.distributions(*inputs, before)
                sampled = distribution.sample()
                log_probability = distribution.log_prob(sampled)
            recurrent = _put_state(recurrent, decision_indices, output.state)
            for local, agent_index in enumerate(decision_indices):
                action = int(sampled[local])
                environment_index, side = divmod(agent_index, 2)
                observation = observations[environment_index][side]
                player = observation["team_state"][0][
                    "designated_possession_player"
                ]
                if 0 <= player < 11 and observation["teams"][0][player]["is_active"]:
                    decisions[environment_index][side][player] = action
                sequence_start = len(pending[agent_index]) % sequence_length == 0
                saved_state = (
                    RecurrentState(
                        before.entities[local:local + 1].detach().cpu().half(),
                        before.global_state[local:local + 1].detach().cpu().half(),
                    )
                    if sequence_start else None
                )
                step_data[environment_index][agent_index] = (
                    frames[local], saved_state, action,
                    float(log_probability[local]), float(output.value[local]),
                )
        for environment_index in slots:
            future = executor.submit(
                environment.step_decision_one,
                environment_index,
                decisions[environment_index],
            )
            inflight[future] = (environment_index, step_data[environment_index])

    def accept(future: Future) -> None:
        nonlocal recurrent, simulated_steps
        expected_index, step_data = inflight.pop(future)
        environment_index, current, result, ended, info = future.result()
        if environment_index != expected_index:
            raise RuntimeError("asynchronous environment returned the wrong slot")
        observations[environment_index] = current
        simulated_steps += 1
        for side in range(2):
            agent_index = 2 * environment_index + side
            if info["action_applied"][side]:
                frame, before, action, log_probability, value = step_data[agent_index]
                pending[agent_index].append(
                    Transition(
                        compact_frame(frame),
                        compact_frame(tensorize(current[side])),
                        before,
                        action,
                        log_probability,
                        value,
                    )
                )
        if ended:
            completed_segments.append((int(result), info["segment_steps"]))
            for side, side_result in ((0, result), (1, -result)):
                agent_index = 2 * environment_index + side
                transitions = pending[agent_index]
                if not transitions:
                    if side_result:
                        raise RuntimeError(
                            "a scored self-play segment has no policy decision"
                        )
                else:
                    transitions[-1].next_frame = compact_frame(
                        tensorize(current[side])
                    )
                    if side_result:
                        transitions[-1].terminal_class = (
                            1 if side_result == 1 else 2
                        )
                    retained.append(Segment(transitions, int(side_result)))
                pending[agent_index] = []
                recurrent = _clear_state(recurrent, agent_index)
            if info["match_done"]:
                completed_matches.append(tuple(info["goals"]))
                reset_slots.add(environment_index)
            else:
                ready.add(environment_index)
        else:
            ready.add(environment_index)

    if progress is not None:
        progress(f"asynchronous start matches=0/{games} active={initial_count}")
    with ThreadPoolExecutor(max_workers=environment.count) as executor:
        while len(completed_matches) < games:
            if reset_slots and inflight:
                done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    accept(future)
                continue
            if reset_slots and not inflight:
                for index in sorted(reset_slots):
                    if started_matches < games:
                        observations[index] = environment.reset_one(index)
                        started_matches += 1
                        ready.add(index)
                reset_slots.clear()
            submit_ready(executor)
            if inflight:
                done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                done.update(future for future in inflight if future.done())
                for future in done:
                    accept(future)
            if progress is not None and simulated_steps >= next_report:
                progress(
                    f"matches={len(completed_matches)}/{games} "
                    f"engine_steps={simulated_steps} inflight={len(inflight)} "
                    f"segments={len(completed_segments)} "
                    f"scored={sum(result != 0 for result, _ in completed_segments)}"
                )
                next_report += report_interval

    if progress is not None:
        progress(
            f"asynchronous complete matches={len(completed_matches)}/{games} "
            f"engine_steps={simulated_steps} segments={len(completed_segments)} "
            f"scored={sum(result != 0 for result, _ in completed_segments)}"
        )

    return (
        pack_rollout(retained, sequence_length),
        completed_segments,
        completed_matches,
    )


def update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 4,
    batch_size: int = 256,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.001,
    transition_coefficient: float = 0.1,
    action_value_coefficient: float = 0.1,
    control_coefficient: float = 0.05,
    space_coefficient: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Optimize PPO, EPV, and engine-grounded auxiliary predictions together."""

    valid_advantages = rollout.advantages[rollout.policy_valid]
    normalized_advantages = torch.zeros_like(rollout.advantages)
    if valid_advantages.numel():
        normalized_advantages[rollout.policy_valid] = (
            valid_advantages - valid_advantages.mean()
        ) / (valid_advantages.std(unbiased=False) + 1e-8)
    sequence_count, sequence_length = rollout.encoded.shape[:2]
    sequences_per_batch = max(batch_size // sequence_length, 1)
    batch_count = math.ceil(sequence_count / sequences_per_batch)
    report_interval = max(batch_count // 4, 1)
    names = (
        "policy", "value", "entropy", "kl", "clip_fraction",
        "outcome", "terminal", "latent", "action_value", "control", "space",
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
            policy_valid = _batch_tensor(rollout.policy_valid, indices, device)
            terminal_supervision = _batch_tensor(
                rollout.terminal_supervision, indices, device
            )
            actions = _batch_tensor(rollout.actions, indices, device)
            old_log_probability = _batch_tensor(
                rollout.old_log_probabilities, indices, device
            )
            advantages = _batch_tensor(normalized_advantages, indices, device)
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
            output = policy.sequence(
                encoded, own, opponent, context,
                anchor_indices, opponent_anchor_indices,
                RecurrentState(initial_entity_state, initial_global_state),
                valid,
            )
            distribution = Categorical(logits=output.logits)
            log_probability = distribution.log_prob(actions)
            log_ratio = log_probability - old_log_probability
            ratio = log_ratio.exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            if policy_valid.any():
                policy_loss = -torch.minimum(unclipped, clipped)[policy_valid].mean()
                value_loss = nn.functional.mse_loss(
                    output.value[policy_valid], returns[policy_valid]
                )
                entropy = distribution.entropy()[policy_valid].mean()
                approximate_kl = (
                    (ratio - 1.0) - log_ratio
                )[policy_valid].mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > clip_ratio
                )[policy_valid].float().mean()
            else:
                zero = output.value.sum() * 0.0
                policy_loss = value_loss = entropy = approximate_kl = zero
                clip_fraction = zero

            action_index = actions.unsqueeze(-1)
            selected_outcome = output.predicted_outcome.gather(
                2, action_index[..., None].expand(-1, -1, 1, output.predicted_outcome.shape[-1])
            ).squeeze(2)
            selected_terminal = output.terminal_logits.gather(
                2, action_index[..., None].expand(-1, -1, 1, 3)
            ).squeeze(2)
            selected_latent = output.predicted_latent.gather(
                2, action_index[..., None].expand(-1, -1, 1, output.predicted_latent.shape[-1])
            ).squeeze(2)
            selected_action_value = output.action_values.gather(2, action_index).squeeze(-1)
            target_outcome = policy.observed_outcome(
                next_own.reshape(-1, 11, next_own.shape[-1]),
                next_opponent.reshape(-1, 11, next_opponent.shape[-1]),
                next_context.reshape(-1, next_context.shape[-1]),
                next_anchor_indices.reshape(-1),
            ).reshape(*valid.shape, -1)
            outcome_loss = nn.functional.mse_loss(selected_outcome[valid], target_outcome[valid])
            if terminal_supervision.any():
                terminal_loss = nn.functional.cross_entropy(
                    selected_terminal[terminal_supervision],
                    terminal_classes[terminal_supervision],
                )
            else:
                terminal_loss = selected_terminal.sum() * 0.0
            latent_valid = valid[:, :-1] & valid[:, 1:]
            latent_valid &= terminal_classes[:, :-1] == 0
            if latent_valid.any():
                latent_loss = nn.functional.smooth_l1_loss(
                    selected_latent[:, :-1][latent_valid],
                    output.latent[:, 1:].detach()[latent_valid],
                )
            else:
                latent_loss = selected_latent.sum() * 0.0
            if policy_valid.any():
                action_value_loss = nn.functional.mse_loss(
                    selected_action_value[policy_valid], returns[policy_valid]
                )
            else:
                action_value_loss = selected_action_value.sum() * 0.0

            players = torch.cat((own, opponent), dim=2)
            reach_target = players[..., 20]
            reach_valid = valid[..., None] & (players[..., PLAYER_ACTIVE_INDEX] > 0.5)
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
                policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
                + transition_coefficient * (outcome_loss + terminal_loss + latent_loss)
                + action_value_coefficient * action_value_loss
                + control_coefficient * control_loss
                + space_coefficient * space_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is NaN or Inf")
            optimizer.step()

            metrics = {
                "policy": policy_loss.item(), "value": value_loss.item(),
                "entropy": entropy.item(), "kl": approximate_kl.item(),
                "clip_fraction": clip_fraction.item(), "outcome": outcome_loss.item(),
                "terminal": terminal_loss.item(), "latent": latent_loss.item(),
                "action_value": action_value_loss.item(), "control": control_loss.item(),
                "space": space_loss.item(),
            }
            policy_count = int(policy_valid.sum())
            metric_weights = {
                "policy": policy_count, "value": policy_count,
                "entropy": policy_count, "kl": policy_count,
                "clip_fraction": policy_count, "outcome": int(valid.sum()),
                "terminal": int(terminal_supervision.sum()),
                "latent": int(latent_valid.sum()),
                "action_value": policy_count, "control": int(reach_valid.sum()),
                "space": int(space_valid.sum()),
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
                f"policy={epoch_totals['policy'] / max(epoch_weights['policy'], 1):.4f} "
                f"value={epoch_totals['value'] / max(epoch_weights['value'], 1):.4f} "
                f"transition={sum(epoch_totals[name] / max(epoch_weights[name], 1) for name in ('outcome', 'terminal', 'latent')):.4f} "
                f"q={epoch_totals['action_value'] / max(epoch_weights['action_value'], 1):.4f} "
                f"entropy={epoch_totals['entropy'] / max(epoch_weights['entropy'], 1):.4f}"
            )
    return {name: totals[name] / max(weights[name], 1) for name in names}
