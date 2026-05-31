import numpy as np


def compute_hard_barrier_distance(v_ego, v_front):
    """
    Short-horizon hard safety barrier.

    This is intentionally less conservative than a full stopping-distance
    barrier. It represents the minimum short-term recoverable distance.

    Used only by CBF.
    """

    min_gap = 3.0
    closing_speed = max(0.0, v_ego - v_front)

    hard_distance = (
        min_gap
        + 0.2 * v_ego
        + 0.4 * closing_speed
    )

    return hard_distance


def cbf_three_vehicle_filter(a_nom, state, dt=0.1):
    """
    CBF-inspired one-step safety filter using the hard barrier.

    PPO outputs a_nom.
    CBF only modifies a_nom if the predicted next-step distance violates
    the hard safety barrier.

    Soft barrier is handled by reward, not by CBF.
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

    candidate_actions = np.linspace(a_min, a_max, 81)

    safe_actions = []

    for a in candidate_actions:
        v_ego_next = max(0.0, v_ego + a * dt)
        v_ego_next = min(v_ego_next, speed_limit)

        v_front_next = v_front

        distance_next = distance_front + (v_front_next - v_ego_next) * dt

        hard_distance_next = compute_hard_barrier_distance(
            v_ego_next,
            v_front_next,
        )

        if distance_next >= hard_distance_next + 0.5:
            safe_actions.append(a)

    if len(safe_actions) == 0:
        return a_min

    safe_actions = np.array(safe_actions)

    best_action = safe_actions[np.argmin((safe_actions - a_nom) ** 2)]

    return float(np.clip(best_action, a_min, a_max))