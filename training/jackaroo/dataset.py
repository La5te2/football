"""Match-level trajectory storage and deterministic dataset partitioning."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
from typing import Any, Sequence


Observation = dict[str, Any]
ODA_HORIZON = 10
ODA_STABLE_STEPS = 2


@dataclass
class Episode:
    observations: list[Observation]
    outcome: int


@dataclass(frozen=True)
class DatasetSizes:
    train: int
    validation: int
    test: int


class EpisodeDataset:
    """Keep complete matches in disjoint train, validation, and test splits."""

    _split_cycle = (
        "train", "validation", "test", "train", "train",
        "train", "train", "train", "train", "train",
    )

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("episode dataset capacity must be positive")
        self.capacity = capacity
        self._base: list[tuple[str, Episode]] = []
        self._recent: deque[tuple[str, Episode]] = deque()
        self._outcome_counts = [0, 0, 0]

    def add(self, episodes: Sequence[Episode], *, base: bool = False) -> None:
        """Assign each complete match to one stable outcome-stratified split."""

        for episode in episodes:
            if episode.outcome not in (0, 1, 2):
                raise ValueError("episode outcome must be win, draw, or loss")
            count = self._outcome_counts[episode.outcome]
            split = self._split_cycle[count % len(self._split_cycle)]
            self._outcome_counts[episode.outcome] += 1
            record = (split, episode)
            if base:
                self._base.append(record)
            else:
                self._recent.append(record)
        while len(self._base) + len(self._recent) > self.capacity:
            if self._recent:
                self._recent.popleft()
            elif self._base:
                self._base.pop(0)

    def split(self, name: str) -> list[Episode]:
        if name not in ("train", "validation", "test"):
            raise ValueError("unknown episode split")
        return [
            episode
            for split, episode in (*self._base, *self._recent)
            if split == name
        ]

    @property
    def train(self) -> list[Episode]:
        return self.split("train")

    @property
    def validation(self) -> list[Episode]:
        return self.split("validation")

    @property
    def test(self) -> list[Episode]:
        return self.split("test")

    def sizes(self) -> DatasetSizes:
        return DatasetSizes(
            train=len(self.train),
            validation=len(self.validation),
            test=len(self.test),
        )

    def __len__(self) -> int:
        return len(self._base) + len(self._recent)

    def state_dict(self) -> dict[str, Any]:
        """Return a tensor-saveable representation of every stored match."""

        def encode(records: Sequence[tuple[str, Episode]]) -> list[dict[str, Any]]:
            return [
                {
                    "split": split,
                    "outcome": episode.outcome,
                    "observations": episode.observations,
                }
                for split, episode in records
            ]

        return {
            "capacity": self.capacity,
            "outcome_counts": list(self._outcome_counts),
            "base": encode(self._base),
            "recent": encode(self._recent),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore match partitions without assigning new split identities."""

        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"dataset capacity {state['capacity']} does not match {self.capacity}"
            )

        def decode(records: Sequence[dict[str, Any]]) -> list[tuple[str, Episode]]:
            result = []
            for record in records:
                split = record["split"]
                if split not in ("train", "validation", "test"):
                    raise ValueError("dataset contains an invalid split")
                result.append(
                    (
                        split,
                        Episode(
                            observations=record["observations"],
                            outcome=int(record["outcome"]),
                        ),
                    )
                )
            return result

        self._base = decode(state["base"])
        self._recent = deque(decode(state["recent"]))
        self._outcome_counts = [int(value) for value in state["outcome_counts"]]
        if len(self._outcome_counts) != 3 or len(self) > self.capacity:
            raise ValueError("dataset state is inconsistent")


def _swap_team(team: int) -> int:
    return 1 - team if team in (0, 1) else team


def _mirror_vector(value: Sequence[float]) -> list[float] | tuple[float, ...]:
    result = list(value)
    if len(result) >= 2:
        result[0] = -result[0]
        result[1] = -result[1]
    return tuple(result) if isinstance(value, tuple) else result


def mirror_observation(observation: Observation) -> Observation:
    """Exchange teams and rotate one canonical observation by 180 degrees."""

    mirrored = copy.deepcopy(observation)
    for name in ("ball_position", "ball_velocity", "ball_rotation"):
        mirrored[name] = _mirror_vector(observation[name])
    teams = []
    for team in reversed(observation["teams"]):
        mirrored_team = []
        for player in team:
            player_copy = copy.deepcopy(player)
            for name in ("position", "velocity", "facing"):
                player_copy[name] = _mirror_vector(player[name])
            player_copy["formation_position"] = _mirror_vector(
                player["formation_position"]
            )
            player_copy["dynamic_formation_position"] = _mirror_vector(
                player["dynamic_formation_position"]
            )
            mirrored_team.append(player_copy)
        teams.append(mirrored_team)
    mirrored["teams"] = tuple(teams) if isinstance(observation["teams"], tuple) else teams
    team_state = copy.deepcopy(list(reversed(observation["team_state"])))
    mirrored["team_state"] = (
        tuple(team_state) if isinstance(observation["team_state"], tuple) else team_state
    )
    for team in mirrored["team_state"]:
        team["offside_trap_x"] = -team["offside_trap_x"]
    mirrored["goals"] = tuple(reversed(observation["goals"]))
    for name in ("set_piece_team", "ball_owned_team", "last_touch_team"):
        mirrored[name] = _swap_team(observation[name])
    sticky = []
    for actions in observation["sticky_actions"]:
        actions = list(actions)
        actions[:8] = actions[4:8] + actions[:4]
        sticky.append(tuple(actions))
    mirrored["sticky_actions"] = tuple(sticky)
    return mirrored


def attention_targets(
    observations: Sequence[Observation],
    horizon: int = ODA_HORIZON,
    stable_steps: int = ODA_STABLE_STEPS,
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Build confidence-weighted offensive and defensive possession labels."""

    offense: list[int] = []
    defense: list[int] = []
    offense_weight: list[float] = []
    defense_weight: list[float] = []

    def owner(observation: Observation) -> tuple[int, int, float]:
        team = observation["ball_owned_team"]
        player = observation["ball_owned_player"]
        if team in (0, 1) and player in range(11) and observation["is_in_play"]:
            source = observation["teams"][team][player]
            confirmed = (
                observation["last_touch_team"] == team
                and observation["last_touch_player"] == player
            )
            own_time = observation["team_state"][team]["time_to_ball_ms"]
            other_time = observation["team_state"][1 - team]["time_to_ball_ms"]
            time_confidence = max(
                0.5, min(1.0, 1.0 + (other_time - own_time) / 2000.0)
            )
            possession_confidence = (
                1.0 if confirmed or source["possession_duration_ms"] > 100 else 0.8
            )
            confidence = possession_confidence * time_confidence
            return team, player, confidence
        return -1, -1, 0.0

    for index, current in enumerate(observations[:-1]):
        current_team, _, current_confidence = owner(current)
        target_player = -1
        target_confidence = 0.0
        run = 0
        previous = -1
        end = min(len(observations), index + horizon + 1)
        for future in observations[index:end]:
            team, player, confidence = owner(future)
            if team == current_team and player in range(11):
                run = run + 1 if player == previous else 1
                previous = player
                if run >= stable_steps:
                    target_player = player
                    target_confidence = confidence
                    break
            elif team in (0, 1) and team != current_team:
                break
            else:
                run = 0
                previous = -1
        stable = target_player in range(11)
        confidence = min(current_confidence, target_confidence) if stable else 0.0
        offense.append(target_player if current_team == 0 else -100)
        defense.append(target_player if current_team == 1 else -100)
        offense_weight.append(confidence if current_team == 0 else 0.0)
        defense_weight.append(confidence if current_team == 1 else 0.0)
    return offense, defense, offense_weight, defense_weight
