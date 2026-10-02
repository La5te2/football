"""Generate a reusable compressed TamakEri self-play dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from .attention import TensorFrame, tensorize
from .env import SelfPlayVectorFootballEnv
from .dataset import DatasetWriter
from .ppo import Segment, Transition, compact_frame
from .tamakeri import TAMAKERI_WEIGHTS, TamakEriTeacher


def _status(message: str) -> None:
    print(f"status={message}", flush=True)


def collect(
    environment: SelfPlayVectorFootballEnv,
    teacher: TamakEriTeacher,
    writer: DatasetWriter,
    games: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Run teacher self-play once and stream every completed segment to disk."""

    first_flight_size = min(environment.count, games)
    observations = environment.reset(first_flight_size)
    teacher.reset()
    pending: list[list[Transition]] = [[] for _ in range(2 * environment.count)]
    completed_segments: list[tuple[int, int]] = []
    completed_matches: list[tuple[int, int]] = []
    simulated_steps = flight = supervised_steps = 0
    report_interval = max(500 * environment.count, 1)
    next_report = report_interval

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
            step_data: dict[int, tuple[TensorFrame, int, bool]] = {}
            if frames:
                teacher_decisions = teacher.decide(teacher_observations, agent_indices)
                for frame, agent_index, teacher_decision in zip(
                    frames, agent_indices, teacher_decisions
                ):
                    environment_index, side = divmod(agent_index, 2)
                    decisions[environment_index][side] = teacher_decision.actions
                    supervised = teacher_decision.action is not None
                    action = teacher_decision.action if supervised else 0
                    supervised_steps += int(supervised)
                    step_data[agent_index] = (frame, action, supervised)

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
                        frame, action, supervised = step_data[agent_index]
                        pending[agent_index].append(
                            Transition(
                                frame=compact_frame(frame),
                                next_frame=compact_frame(
                                    tensorize(next_observations[environment_index][side])
                                ),
                                state=None,
                                action=action,
                                old_log_probability=0.0,
                                old_value=0.0,
                                action_supervised=supervised,
                            )
                        )
                if segment_done[environment_index]:
                    result = int(results[environment_index])
                    completed_segments.append(
                        (result, infos[environment_index]["segment_steps"])
                    )
                    for side, side_result in ((0, result), (1, -result)):
                        agent_index = 2 * environment_index + side
                        if pending[agent_index]:
                            pending[agent_index][-1].next_frame = compact_frame(
                                tensorize(next_observations[environment_index][side])
                            )
                            if side_result:
                                pending[agent_index][-1].terminal_class = (
                                    1 if side_result == 1 else 2
                                )
                            writer.append_segment(
                                Segment(pending[agent_index], side_result)
                            )
                        pending[agent_index] = []
                    if infos[environment_index]["match_done"]:
                        completed_matches.append(
                            tuple(infos[environment_index]["goals"])
                        )
                        writer.set_matches(len(completed_matches))
                        active[environment_index] = False
            observations = next_observations
            if progress is not None and simulated_steps >= next_report:
                progress(
                    f"flight={flight} matches={len(completed_matches)}/{games} "
                    f"engine_steps={simulated_steps} active={sum(active)}/{flight_size} "
                    f"segments={len(completed_segments)} labels={supervised_steps}"
                )
                next_report += report_interval
        if progress is not None:
            progress(
                f"flight={flight} complete matches={len(completed_matches)}/{games} "
                f"segments={len(completed_segments)} labels={supervised_steps}"
            )
    return completed_segments, completed_matches


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--flight", type=int, default=16)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/tamakeri.h5"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    for name in ("games", "flight", "maximum_steps", "sequence_length"):
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    device = torch.device(
        "cuda"
        if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto" else arguments.device
    )
    temporary = arguments.output.with_name(arguments.output.name + ".tmp")
    _status(
        f"preparation start device={device.type} games={arguments.games} "
        f"flight={arguments.flight} output={arguments.output}"
    )
    environment = SelfPlayVectorFootballEnv(
        arguments.flight, arguments.maximum_steps, arguments.seed
    )
    teacher = TamakEriTeacher(
        2 * arguments.flight,
        arguments.maximum_steps,
        device,
        TAMAKERI_WEIGHTS,
    )
    with DatasetWriter(temporary, arguments.sequence_length) as writer:
        segments, matches = collect(
            environment,
            teacher,
            writer,
            arguments.games,
            progress=lambda message: _status(f"preparation {message}"),
        )
        transitions = writer.transitions
        sequences = writer.sequences
    temporary.replace(arguments.output)
    _status(
        f"preparation complete matches={len(matches)} segments={len(segments)} "
        f"scored={sum(result != 0 for result, _ in segments)} "
        f"tails={sum(result == 0 for result, _ in segments)} "
        f"transitions={transitions} sequences={sequences} "
        f"output={arguments.output}"
    )


if __name__ == "__main__":
    main()
