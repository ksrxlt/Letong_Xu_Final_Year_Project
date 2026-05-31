import numpy as np


def cbf_three_vehicle_filter(a_nom, state, dt=0.1):
    """
    One-step predictive CBF-inspired safety filter.

    Current cut-in state:
        [
            v_ego,
            v_front_effective,
            distance_front,
            relative_velocity_front,
            cut_in_happened,
            remaining_distance
        ]

    PPO outputs a_nom.
    CBF checks whether a_nom would violate the next-step safety distance.
    """

    (
        v_ego,
        v_front,
        distance_front,
        relative_velocity_front,
        cut_in_happened,
        remaining_distance,
    ) = state

    a_min = -4.0
    a_max = 2.0
    speed_limit = 130.0 / 3.6

    a_nom = float(np.clip(a_nom, a_min, a_max))

    candidate_actions = np.linspace(a_min, a_max, 61)
    safe_actions = []

    for a in candidate_actions:
        v_ego_next = max(0.0, v_ego + a * dt)
        v_ego_next = min(v_ego_next, speed_limit)

        v_front_next = v_front

        distance_next = distance_front + (v_front_next - v_ego_next) * dt
        safe_distance_next = 5.0 + 0.5 * v_ego_next

        # Add a small buffer so CBF is not too weak
        if distance_next >= safe_distance_next + 1.0:
            safe_actions.append(a)

    if len(safe_actions) == 0:
        return a_min

    safe_actions = np.array(safe_actions)

    # Choose the safe action closest to PPO nominal action
    best_action = safe_actions[np.argmin((safe_actions - a_nom) ** 2)]

    return float(np.clip(best_action, a_min, a_max))