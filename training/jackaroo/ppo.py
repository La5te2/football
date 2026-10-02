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
    sequence_valid: torch.Tensor
    valid: torch.Tensor
    policy_valid: torch.Tensor
    terminal_supervision: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminal_classes: torch.Tensor
    action_supervision: torch.Tensor
    reward_total: float
    transition_count: int
    burn_in: int


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
    left_padding: list[int],
    length: int,
    next_frame: bool,
) -> tuple[torch.Tensor, ...]:
    fields: list[torch.Tensor] = []
    for field in range(len(TensorFrame._fields)):
        sequences = []
        for chunk, prefix in zip(chunks, left_padding):
            values = [
                (transition.next_frame if next_frame else transition.frame)[field]
                for transition in chunk
            ]
            value = torch.stack(values)
            suffix = length - prefix - len(values)
            padding_shape = value.shape[1:]
            if prefix:
                value = torch.cat(
                    (torch.zeros((prefix, *padding_shape), dtype=value.dtype), value)
                )
            if suffix:
                value = torch.cat(
                    (value, torch.zeros((suffix, *padding_shape), dtype=value.dtype))
                )
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


def pack_rollout(
    segments: list[Segment], sequence_length: int, burn_in: int = 0
) -> Rollout:
    if sequence_length <= 0 or burn_in not in (0, sequence_length):
        raise ValueError("burn-in must be zero or equal to the sequence length")
    chunks: list[list[Transition]] = []
    learning_chunks: list[list[Transition]] = []
    left_padding: list[int] = []
    chunk_results: list[int] = []
    chunk_ends_segment: list[bool] = []
    for segment in segments:
        for start in range(0, len(segment.transitions), sequence_length):
            history_start = max(0, start - burn_in)
            history_length = start - history_start
            chunks.append(
                segment.transitions[
                    history_start:start + sequence_length
                ]
            )
            learning_chunks.append(
                segment.transitions[start:start + sequence_length]
            )
            left_padding.append(burn_in - history_length)
            chunk_results.append(segment.result)
            chunk_ends_segment.append(
                start + sequence_length >= len(segment.transitions)
            )
    if not chunks:
        raise RuntimeError("rollout contains no retained segment")

    packed_length = burn_in + sequence_length
    current = _stack_padded_frames(
        chunks, left_padding, packed_length, False
    )
    following = _stack_padded_frames(
        chunks, left_padding, packed_length, True
    )
    sequence_count = len(chunks)
    sequence_valid = torch.zeros(sequence_count, packed_length, dtype=torch.bool)
    valid = torch.zeros(sequence_count, packed_length, dtype=torch.bool)
    policy_valid = torch.zeros(sequence_count, packed_length, dtype=torch.bool)
    terminal_supervision = torch.zeros(
        sequence_count, packed_length, dtype=torch.bool
    )
    actions = torch.zeros(sequence_count, packed_length, dtype=torch.long)
    old_log_probabilities = torch.zeros(sequence_count, packed_length)
    old_values = torch.zeros(sequence_count, packed_length)
    returns = torch.zeros(sequence_count, packed_length)
    terminal_classes = torch.zeros(
        sequence_count, packed_length, dtype=torch.long
    )
    action_supervision = torch.zeros(
        sequence_count, packed_length, dtype=torch.bool
    )
    initial_entities = []
    initial_globals = []
    transition_count = 0
    for index, (chunk, learning, result, ends_segment, prefix) in enumerate(
        zip(
            chunks,
            learning_chunks,
            chunk_results,
            chunk_ends_segment,
            left_padding,
        )
    ):
        sequence_valid[index, prefix:prefix + len(chunk)] = True
        count = len(learning)
        transition_count += count
        begin = burn_in
        end = begin + count
        valid[index, begin:end] = True
        if result:
            policy_valid[index, begin:end] = True
        terminal_supervision[index, begin:end] = True
        if result == 0 and ends_segment:
            terminal_supervision[index, end - 1] = False
        actions[index, begin:end] = torch.tensor(
            [item.action for item in learning]
        )
        old_log_probabilities[index, begin:end] = torch.tensor(
            [item.old_log_probability for item in learning]
        )
        old_values[index, begin:end] = torch.tensor(
            [item.old_value for item in learning]
        )
        returns[index, begin:end] = float(result)
        terminal_classes[index, begin:end] = torch.tensor(
            [item.terminal_class for item in learning]
        )
        action_supervision[index, begin:end] = torch.tensor(
            [item.action_supervised for item in learning]
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
        sequence_valid,
        valid,
        policy_valid,
        terminal_supervision,
        actions,
        old_log_probabilities,
        old_values,
        returns - old_values,
        returns,
        terminal_classes,
        action_supervision,
        float(sum(segment.result for segment in segments)),
        transition_count,
        burn_in,
    )


def collect_self_play_rollout(
    environment: SelfPlayVectorFootballEnv,
    policy: ActorCritic,
    games: int,
    sequence_length: int = 32,
    burn_in: int = 32,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    Rollout,
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Collect full matches with independently scheduled native simulations."""

    if games <= 0 or sequence_length <= 0 or burn_in < 0:
        raise ValueError(
            "rollout games and sequence length must be positive and burn-in nonnegative"
        )
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
        pack_rollout(retained, sequence_length, burn_in),
        completed_segments,
        completed_matches,
    )


def update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    epochs: int = 4,
    structure_epochs: int = 1,
    batch_size: int = 2048,
    clip_ratio: float = 0.2,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.001,
    transition_coefficient: float = 0.1,
    action_value_coefficient: float = 0.1,
    control_coefficient: float = 0.05,
    space_coefficient: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """Run PPO and football-structure optimization as separate phases."""

    if epochs <= 0 or structure_epochs <= 0 or batch_size <= 0:
        raise ValueError("optimization epochs and batch size must be positive")
    if clip_ratio <= 0.0 or value_clip <= 0.0:
        raise ValueError("policy and value clipping ranges must be positive")

    valid_advantages = rollout.advantages[rollout.policy_valid]
    normalized_advantages = torch.zeros_like(rollout.advantages)
    if valid_advantages.numel():
        normalized_advantages[rollout.policy_valid] = (
            valid_advantages - valid_advantages.mean()
        ) / (valid_advantages.std(unbiased=False) + 1e-8)
    sequence_count = rollout.encoded.shape[0]
    learning_length = max(int(rollout.valid.sum(dim=1).max()), 1)
    sequences_per_batch = max(batch_size // learning_length, 1)
    batch_count = math.ceil(sequence_count / sequences_per_batch)
    report_interval = max(batch_count // 4, 1)
    names = (
        "policy", "value", "entropy", "kl", "clip_fraction",
        "value_clip_fraction",
        "outcome", "terminal", "latent", "action_value", "control", "space",
    )
    totals = {name: 0.0 for name in names}
    weights = {name: 0 for name in names}
    device = next(policy.parameters()).device
    policy.train()

    def load(indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "encoded": _batch_tensor(rollout.encoded, indices, device),
            "own": _batch_tensor(rollout.own, indices, device),
            "opponent": _batch_tensor(rollout.opponent, indices, device),
            "context": _batch_tensor(rollout.context, indices, device),
            "anchor": _batch_tensor(rollout.anchor_indices, indices, device),
            "opponent_anchor": _batch_tensor(
                rollout.opponent_anchor_indices, indices, device
            ),
            "initial_entities": _batch_tensor(
                rollout.initial_entity_state, indices, device
            ),
            "initial_global": _batch_tensor(
                rollout.initial_global_state, indices, device
            ),
            "sequence_valid": _batch_tensor(
                rollout.sequence_valid, indices, device
            ),
            "valid": _batch_tensor(rollout.valid, indices, device),
            "policy_valid": _batch_tensor(
                rollout.policy_valid, indices, device
            ),
            "terminal_supervision": _batch_tensor(
                rollout.terminal_supervision, indices, device
            ),
            "actions": _batch_tensor(rollout.actions, indices, device),
            "old_log_probability": _batch_tensor(
                rollout.old_log_probabilities, indices, device
            ),
            "old_values": _batch_tensor(rollout.old_values, indices, device),
            "advantages": _batch_tensor(
                normalized_advantages, indices, device
            ),
            "returns": _batch_tensor(rollout.returns, indices, device),
            "terminal_classes": _batch_tensor(
                rollout.terminal_classes, indices, device
            ),
            "next_own": _batch_tensor(rollout.next_own, indices, device),
            "next_opponent": _batch_tensor(
                rollout.next_opponent, indices, device
            ),
            "next_context": _batch_tensor(
                rollout.next_context, indices, device
            ),
            "next_anchor": _batch_tensor(
                rollout.next_anchor_indices, indices, device
            ),
        }

    def forward(batch: dict[str, torch.Tensor]):
        return policy.sequence(
            batch["encoded"], batch["own"], batch["opponent"],
            batch["context"], batch["anchor"], batch["opponent_anchor"],
            RecurrentState(
                batch["initial_entities"], batch["initial_global"]
            ),
            batch["sequence_valid"],
            rollout.burn_in,
        )

    def accumulate(
        metrics: dict[str, float],
        metric_weights: dict[str, int],
        epoch_totals: dict[str, float],
        epoch_weights: dict[str, int],
    ) -> None:
        for name, metric in metrics.items():
            weight = metric_weights[name]
            totals[name] += metric * weight
            weights[name] += weight
            epoch_totals[name] += metric * weight
            epoch_weights[name] += weight

    for epoch in range(1, structure_epochs + 1):
        epoch_totals = {name: 0.0 for name in names}
        epoch_weights = {name: 0 for name in names}
        batches = torch.randperm(sequence_count).split(sequences_per_batch)
        for batch_number, indices in enumerate(batches, start=1):
            batch = load(indices)
            output = forward(batch)
            valid = batch["valid"]
            policy_valid = batch["policy_valid"]
            action_index = batch["actions"].unsqueeze(-1)
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
                batch["next_own"].reshape(
                    -1, 11, batch["next_own"].shape[-1]
                ),
                batch["next_opponent"].reshape(
                    -1, 11, batch["next_opponent"].shape[-1]
                ),
                batch["next_context"].reshape(
                    -1, batch["next_context"].shape[-1]
                ),
                batch["next_anchor"].reshape(-1),
            ).reshape(*valid.shape, -1)
            outcome_loss = nn.functional.mse_loss(
                selected_outcome[valid], target_outcome[valid]
            )
            terminal_supervision = batch["terminal_supervision"]
            if terminal_supervision.any():
                terminal_loss = nn.functional.cross_entropy(
                    selected_terminal[terminal_supervision],
                    batch["terminal_classes"][terminal_supervision],
                )
            else:
                terminal_loss = selected_terminal.sum() * 0.0
            latent_valid = valid[:, :-1] & valid[:, 1:]
            latent_valid &= batch["terminal_classes"][:, :-1] == 0
            if latent_valid.any():
                latent_loss = nn.functional.smooth_l1_loss(
                    selected_latent[:, :-1][latent_valid],
                    output.latent[:, 1:].detach()[latent_valid],
                )
            else:
                latent_loss = selected_latent.sum() * 0.0
            if policy_valid.any():
                action_value_loss = nn.functional.mse_loss(
                    selected_action_value[policy_valid],
                    batch["returns"][policy_valid],
                )
            else:
                action_value_loss = selected_action_value.sum() * 0.0

            players = torch.cat((batch["own"], batch["opponent"]), dim=2)
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
                batch["own"].reshape(-1, 11, batch["own"].shape[-1]),
                batch["opponent"].reshape(
                    -1, 11, batch["opponent"].shape[-1]
                ),
            ).reshape(*valid.shape, 2, -1)
            possession = torch.stack(
                (
                    batch["context"][..., OWN_POSSESSION_INDEX] > 0.5,
                    batch["context"][..., OPPONENT_POSSESSION_INDEX] > 0.5,
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
                transition_coefficient
                * (outcome_loss + terminal_loss + latent_loss)
                + action_value_coefficient * action_value_loss
                + control_coefficient * control_loss
                + space_coefficient * space_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("structure loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("structure gradient norm is NaN or Inf")
            optimizer.step()

            metrics = {
                "outcome": outcome_loss.item(),
                "terminal": terminal_loss.item(),
                "latent": latent_loss.item(),
                "action_value": action_value_loss.item(),
                "control": control_loss.item(),
                "space": space_loss.item(),
            }
            metric_weights = {
                "outcome": int(valid.sum()),
                "terminal": int(terminal_supervision.sum()),
                "latent": int(latent_valid.sum()),
                "action_value": int(policy_valid.sum()),
                "control": int(reach_valid.sum()),
                "space": int(space_valid.sum()),
            }
            accumulate(metrics, metric_weights, epoch_totals, epoch_weights)
            if progress is not None and (
                batch_number % report_interval == 0
                or batch_number == batch_count
            ):
                progress(
                    f"phase=structure epoch={epoch}/{structure_epochs} "
                    f"batch={batch_number}/{batch_count}"
                )
        if progress is not None:
            progress(
                f"phase=structure epoch={epoch}/{structure_epochs} complete "
                f"transition={sum(epoch_totals[name] / max(epoch_weights[name], 1) for name in ('outcome', 'terminal', 'latent')):.4f} "
                f"q={epoch_totals['action_value'] / max(epoch_weights['action_value'], 1):.4f} "
                f"control={epoch_totals['control'] / max(epoch_weights['control'], 1):.4f} "
                f"space={epoch_totals['space'] / max(epoch_weights['space'], 1):.4f}"
            )

    for epoch in range(1, epochs + 1):
        epoch_totals = {name: 0.0 for name in names}
        epoch_weights = {name: 0 for name in names}
        batches = torch.randperm(sequence_count).split(sequences_per_batch)
        for batch_number, indices in enumerate(batches, start=1):
            batch = load(indices)
            output = forward(batch)
            distribution = Categorical(logits=output.logits)
            log_probability = distribution.log_prob(batch["actions"])
            log_ratio = log_probability - batch["old_log_probability"]
            ratio = log_ratio.exp()
            unclipped = ratio * batch["advantages"]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch[
                "advantages"
            ]
            policy_valid = batch["policy_valid"]
            if policy_valid.any():
                policy_loss = -torch.minimum(
                    unclipped, clipped
                )[policy_valid].mean()
                clipped_value = batch["old_values"] + (
                    output.value - batch["old_values"]
                ).clamp(-value_clip, value_clip)
                value_error = (output.value - batch["returns"]).square()
                clipped_value_error = (
                    clipped_value - batch["returns"]
                ).square()
                value_loss = torch.maximum(
                    value_error, clipped_value_error
                )[policy_valid].mean()
                value_clip_fraction = (
                    (output.value - batch["old_values"]).abs() > value_clip
                )[policy_valid].float().mean()
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
                clip_fraction = value_clip_fraction = zero

            loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is NaN or Inf")
            optimizer.step()

            count = int(policy_valid.sum())
            accumulate(
                {
                    "policy": policy_loss.item(),
                    "value": value_loss.item(),
                    "entropy": entropy.item(),
                    "kl": approximate_kl.item(),
                    "clip_fraction": clip_fraction.item(),
                    "value_clip_fraction": value_clip_fraction.item(),
                },
                {
                    "policy": count, "value": count, "entropy": count,
                    "kl": count, "clip_fraction": count,
                    "value_clip_fraction": count,
                },
                epoch_totals,
                epoch_weights,
            )
            if progress is not None and (
                batch_number % report_interval == 0
                or batch_number == batch_count
            ):
                progress(
                    f"phase=ppo epoch={epoch}/{epochs} "
                    f"batch={batch_number}/{batch_count}"
                )
        if progress is not None:
            progress(
                f"phase=ppo epoch={epoch}/{epochs} complete "
                f"policy={epoch_totals['policy'] / max(epoch_weights['policy'], 1):.4f} "
                f"value={epoch_totals['value'] / max(epoch_weights['value'], 1):.4f} "
                f"entropy={epoch_totals['entropy'] / max(epoch_weights['entropy'], 1):.4f}"
            )

    return {name: totals[name] / max(weights[name], 1) for name in names}
