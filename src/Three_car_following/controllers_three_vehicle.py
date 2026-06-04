import numpy as np


def compute_hard_barrier_distance(v_ego, v_front):
    """
    Hard safety barrier.

    This barrier is stricter than the previous version, but still less
    conservative than the soft comfort barrier.

    It represents a short-horizon safety boundary:
        - larger ego speed requires larger distance
        - larger closing speed requires larger distance

    closing_speed = v_ego - v_front
    """

    min_gap = 4.0
    closing_speed = max(0.0, v_ego - v_front)

    hard_distance = (
        min_gap
        + 0.35 * v_ego
        + 0.80 * closing_speed
    )

    return hard_distance


def cbf_three_vehicle_filter(
    a_nom,
    state,
    dt=0.1,
    gamma=1.0,
    safety_margin=0.0,
):
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

    hard_current = compute_hard_barrier_distance(v_ego, v_front)
    h_current = distance_front - hard_current

    closing_speed = max(0.0, v_ego - v_front)

    # CBF only activates near the hard barrier.
    activation_margin = 5.0 + 0.8 * closing_speed
    if h_current > activation_margin:
        return a_nom

    def is_action_safe(a):
        v_ego_next = max(0.0, v_ego + a * dt)
        v_ego_next = min(v_ego_next, speed_limit)

        v_front_next = v_front
        distance_next = distance_front + (v_front_next - v_ego_next) * dt

        hard_next = compute_hard_barrier_distance(
            v_ego_next,
            v_front_next,
        )

        h_next = distance_next - hard_next

        return h_next >= (1.0 - gamma * dt) * h_current + safety_margin

    if is_action_safe(a_nom):
        return a_nom

    # Brake-only candidate actions.
    candidate_actions = np.linspace(a_min, a_nom, 241)

    safe_actions = []

    for a in candidate_actions:
        if is_action_safe(a):
            safe_actions.append(a)

    if len(safe_actions) == 0:
        return a_min

    safe_actions = np.array(safe_actions)

    # Since CBF is brake-only, choose the largest safe action.
    best_action = safe_actions[np.argmax(safe_actions)]

    return float(np.clip(best_action, a_min, a_max))