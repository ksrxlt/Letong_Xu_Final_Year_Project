def rule_based_controller(state):
    """
    A simple nominal controller.

    It tries to keep a desired distance from the front car.
    Later, this controller can be replaced by an RL policy.
    """
    v_ego, v_front, distance, relative_velocity = state

    desired_distance = 20.0

    if distance > desired_distance + 5.0:
        # Too far from the front car, accelerate
        return 1.0
    elif distance < desired_distance:
        # Too close to the front car, brake
        return -2.0
    else:
        # Distance is acceptable, keep current speed
        return 0.0


def cbf_safety_filter(a_nom, state):
    """
    A simple CBF-inspired safety filter.

    The nominal controller suggests a_nom.
    The safety filter modifies it if the distance is unsafe.
    """
    v_ego, v_front, distance, relative_velocity = state

    safe_distance = 5.0 + 0.5 * v_ego

    # If already too close, force braking
    if distance < safe_distance:
        return min(a_nom, -2.5)

    # If approaching the front car and distance is becoming risky, brake slightly
    if distance < safe_distance + 5.0 and relative_velocity < 0:
        return min(a_nom, -1.0)

    # Otherwise, use nominal action
    return a_nom