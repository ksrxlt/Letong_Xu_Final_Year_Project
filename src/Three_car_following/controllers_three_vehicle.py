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
    gamma=1.5,
    safety_margin=0.2,
):
    """
    CBF safety filter for the three-vehicle following task.

    Input:
        a_nom:
            PPO nominal acceleration.

        state:
            [
                v_ego,
                v_front_effective,
                distance_front,
                relative_velocity_front,
                cut_in_happened,
                remaining_distance
            ]

    Output:
        a_filtered:
            acceleration after CBF filtering.

    CBF idea:
        Define barrier function:

            h = distance_front - hard_barrier

        where h >= 0 means the ego vehicle is outside the hard safety boundary.

        The discrete-time CBF condition is:

            h_next >= (1 - gamma * dt) * h_current

        This means the safety margin is not allowed to decrease too quickly.
        If the ego vehicle approaches the hard barrier too aggressively, the
        filter will modify the PPO action.

    Notes:
        - This is still a simple grid-search implementation.
        - It is easier to understand and debug than solving a QP directly.
        - It chooses the safe action closest to PPO's original action.
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

    # Current barrier value
    hard_current = compute_hard_barrier_distance(v_ego, v_front)
    h_current = distance_front - hard_current

    # Candidate actions.
    # More points gives smoother CBF output.
    candidate_actions = np.linspace(a_min, a_max, 241)

    safe_actions = []

    for a in candidate_actions:
        # Predict next ego speed
        v_ego_next = max(0.0, v_ego + a * dt)
        v_ego_next = min(v_ego_next, speed_limit)

        # Simple one-step prediction:
        # assume front vehicle keeps its current speed within this small dt.
        v_front_next = v_front

        # Predict next distance
        distance_next = distance_front + (v_front_next - v_ego_next) * dt

        # Next hard barrier
        hard_next = compute_hard_barrier_distance(
            v_ego_next,
            v_front_next,
        )

        h_next = distance_next - hard_next

        # Discrete CBF condition
        cbf_condition = h_next >= (1.0 - gamma * dt) * h_current + safety_margin

        if cbf_condition:
            safe_actions.append(a)

    # If no action satisfies the CBF condition, use maximum braking.
    if len(safe_actions) == 0:
        return a_min

    safe_actions = np.array(safe_actions)

    # Pick the safe action closest to PPO nominal action.
    best_action = safe_actions[np.argmin((safe_actions - a_nom) ** 2)]

    return float(np.clip(best_action, a_min, a_max))