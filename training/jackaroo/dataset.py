"""Compressed trajectory dataset generation for Jackaroo pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import torch
import numpy as np

from .attention import TensorFrame

if TYPE_CHECKING:
    from .ppo import Segment


FRAME_FIELDS = (
    "encoded", "own", "opponent", "context", "anchor", "opponent_anchor"
)
NEXT_FRAME_FIELDS = ("own", "opponent", "context", "anchor")


class DatasetWriter:
    """Append complete teacher segments to a compressed sequence dataset."""

    def __init__(self, path: Path, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence length must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sequence_length = sequence_length
        self._file = h5py.File(path, "w")
        self._file.attrs["sequence_length"] = sequence_length
        self._file.attrs["scored_segments"] = 0
        self._file.attrs["censored_segments"] = 0
        self._file.attrs["matches"] = 0
        self._file.attrs["trajectories"] = 0
        self._datasets: dict[str, h5py.Dataset] = {}
        self.transitions = 0
        self.sequences = 0
        self.trajectories = 0

    def _dataset(self, name: str, value: np.ndarray) -> h5py.Dataset:
        dataset = self._datasets.get(name)
        if dataset is None:
            shape = (0, *value.shape)
            maximum = (None, *value.shape)
            chunks = (1, *value.shape)
            dataset = self._file.create_dataset(
                name,
                shape=shape,
                maxshape=maximum,
                chunks=chunks,
                dtype=value.dtype,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            self._datasets[name] = dataset
        return dataset

    def _append(self, values: dict[str, np.ndarray]) -> None:
        index = self.sequences
        for name, value in values.items():
            dataset = self._dataset(name, value)
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self.sequences += 1

    @staticmethod
    def _padded(frames: list[TensorFrame], length: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for field, name in enumerate(FRAME_FIELDS):
            value = torch.stack([frame[field] for frame in frames])
            padding = (length - len(frames), *value.shape[1:])
            if padding[0]:
                value = torch.cat((value, torch.zeros(padding, dtype=value.dtype)))
            if value.is_floating_point():
                value = value.to(torch.float16)
            result[name] = value.numpy()
        return result

    def append_segment(self, segment: "Segment") -> None:
        transitions = segment.transitions
        if not transitions:
            return
        self.transitions += len(transitions)
        trajectory = self.trajectories
        self.trajectories += 1
        attribute = "scored_segments" if segment.result else "censored_segments"
        self._file.attrs.modify(attribute, int(self._file.attrs[attribute]) + 1)
        for start in range(0, len(transitions), self.sequence_length):
            chunk = transitions[start:start + self.sequence_length]
            count = len(chunk)
            current = self._padded([item.frame for item in chunk], self.sequence_length)
            following = self._padded(
                [item.next_frame for item in chunk], self.sequence_length
            )
            values = {name: current[name] for name in FRAME_FIELDS}
            values.update(
                {f"next_{name}": following[name] for name in NEXT_FRAME_FIELDS}
            )
            valid = np.zeros(self.sequence_length, dtype=np.bool_)
            valid[:count] = True
            policy_valid = valid.copy() if segment.result else np.zeros_like(valid)
            terminal_supervision = valid.copy()
            if not segment.result and start + count == len(transitions):
                terminal_supervision[count - 1] = False
            actions = np.zeros(self.sequence_length, dtype=np.uint8)
            actions[:count] = [item.action for item in chunk]
            action_supervision = np.zeros(self.sequence_length, dtype=np.bool_)
            action_supervision[:count] = [
                item.action_supervised for item in chunk
            ]
            terminal_classes = np.zeros(self.sequence_length, dtype=np.uint8)
            terminal_classes[:count] = [item.terminal_class for item in chunk]
            returns = np.zeros(self.sequence_length, dtype=np.int8)
            if segment.result:
                returns[:count] = segment.result
            values.update(
                {
                    "valid": valid,
                    "policy_valid": policy_valid,
                    "terminal_supervision": terminal_supervision,
                    "actions": actions,
                    "action_supervision": action_supervision,
                    "terminal_classes": terminal_classes,
                    "returns": returns,
                    "trajectory": np.asarray(trajectory, dtype=np.int64),
                    "trajectory_end": np.asarray(
                        start + count == len(transitions), dtype=np.bool_
                    ),
                }
            )
            self._append(values)

    def set_matches(self, matches: int) -> None:
        self._file.attrs.modify("matches", matches)

    def close(self) -> None:
        if self._file:
            self._file.attrs["sequences"] = self.sequences
            self._file.attrs["transitions"] = self.transitions
            self._file.attrs["trajectories"] = self.trajectories
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
