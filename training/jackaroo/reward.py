"""Fixed football potential and policy-invariant training reward."""

from __future__ import annotations

from typing import Any


Observation = dict[str, Any]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def football_potential(observation: Observation) -> float:
    """Estimate territorial advantage from the public canonical observation.

    The controlled team always attacks toward positive x. The value combines
    ball territory, current control, and the distance of each team's attack
    from its target goal. It is bounded so shaping cannot dominate goals.
    """

    ball_x = _clip(float(observation["ball_position"][0]), -1.0, 1.0)
    owner = int(observation["ball_owned_team"])
    if owner == 0:
        control = 1.0
    elif owner == 1:
        control = -1.0
    else:
        own = float(observation["team_state"][0]["fading_possession_amount"])
        opponent = float(
            observation["team_state"][1]["fading_possession_amount"]
        )
        control = _clip((own - opponent) / 2.0, -1.0, 1.0)

    own_weight = 0.5 * (1.0 + control)
    opponent_weight = 1.0 - own_weight
    own_progress = 0.5 * (ball_x + 1.0)
    opponent_progress = 0.5 * (1.0 - ball_x)
    threat_balance = (
        own_weight * own_progress * own_progress
        - opponent_weight * opponent_progress * opponent_progress
    )
    return _clip(
        0.60 * threat_balance + 0.25 * control + 0.15 * ball_x,
        -1.0,
        1.0,
    )


def transition_reward(
    previous: Observation,
    current: Observation,
    terminated: bool,
    previous_potential: float,
    potential_scale: float,
    gamma: float = 1.0,
) -> tuple[float, float, dict[str, float]]:
    """Return goal, terminal-result, and potential-based reward components."""

    previous_difference = int(previous["goals"][0]) - int(previous["goals"][1])
    current_difference = int(current["goals"][0]) - int(current["goals"][1])
    goal = float(current_difference - previous_difference)
    result = (
        float((current_difference > 0) - (current_difference < 0))
        if terminated
        else 0.0
    )
    current_potential = 0.0 if terminated else football_potential(current)
    shaping = potential_scale * (
        gamma * current_potential - previous_potential
    )
    components = {
        "goal": goal,
        "result": result,
        "task": goal + result,
        "potential": shaping,
    }
    return sum((goal, result, shaping)), current_potential, components
