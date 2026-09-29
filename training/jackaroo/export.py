"""Export a deployable Jackaroo TorchScript policy from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .attention import CONTEXT_FEATURES, PLAYER_FEATURES
from .network import ActorCritic, POLICY_ARCHITECTURE, POLICY_OBJECTIVE


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != POLICY_ARCHITECTURE:
        raise ValueError(
            f"checkpoint architecture={checkpoint.get('architecture')!r} does not "
            f"match {POLICY_ARCHITECTURE!r}"
        )
    if checkpoint.get("objective") != POLICY_OBJECTIVE:
        raise ValueError(
            f"checkpoint objective={checkpoint.get('objective')!r} does not match "
            f"{POLICY_OBJECTIVE!r}"
        )
    required = ("model", "frame_size", "action_count")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "checkpoint is missing required fields: " + ", ".join(missing)
        )

    policy = ActorCritic(
        int(checkpoint["frame_size"]), int(checkpoint["action_count"])
    )
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()

    history_length = int(checkpoint.get("history_length", 1))
    if history_length <= 0:
        raise ValueError("checkpoint history length must be positive")
    example = (
        torch.zeros(1, history_length, int(checkpoint["frame_size"])),
        torch.zeros(1, history_length, 11, PLAYER_FEATURES),
        torch.zeros(1, history_length, 11, PLAYER_FEATURES),
        torch.zeros(1, history_length, CONTEXT_FEATURES),
        torch.zeros(1, history_length, dtype=torch.long),
        torch.zeros(1, history_length, dtype=torch.long),
    )
    with torch.inference_mode():
        exported = torch.jit.trace(policy, example, strict=True)
        expected = policy(*example)
        actual = exported(*example)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(expected_tensor, actual_tensor)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(arguments.output.name + ".tmp")
    torch.jit.save(exported, temporary)
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
