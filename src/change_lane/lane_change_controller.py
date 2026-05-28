def rule_based_acceleration_controller(state):
    """
    Longitudinal controller for multi-lane scenario.

    State:
        [v_ego, lane_ego, distance_front, relative_velocity_front,
         front_gap_target, rear_gap_target]
    """

    v_ego, lane_ego, distance_front, relative_velocity_front, front_gap_target, rear_gap_target = state

    desired_distance = 20.0

    if distance_front > desired_distance + 8.0:
        return 1.0
    elif distance_front < desired_distance:
        return -2.0
    else:
        return 0.0


def lane_change_decision(state):
    """
    Nominal rule-based lane-change decision.

    This controller only decides whether the ego vehicle wants to change lane.
    It does not check whether the target lane is safe.
    Safety is handled separately by the lane-change safety filter.
    """

    v_ego, lane_ego, distance_front, relative_velocity_front, front_gap_target, rear_gap_target = state

    # Only try to overtake when ego is in lane 0
    if int(lane_ego) == 1:
        return 0

    front_vehicle_close = distance_front < 25.0
    front_vehicle_slower = relative_velocity_front < -2.0

    if front_vehicle_close and front_vehicle_slower:
        return 1

    return 0


def lane_change_safety_filter(lane_change_command, state):
    """
    Safety filter for lane-change decision.

    It blocks lane change if target lane gaps are unsafe.
    """

    if lane_change_command == 0:
        return 0

    v_ego, lane_ego, distance_front, relative_velocity_front, front_gap_target, rear_gap_target = state

    if front_gap_target > 15.0 and rear_gap_target > 10.0:
        return 1

    return 0


def longitudinal_safety_filter(a_nom, state):
    """
    CBF-inspired longitudinal safety filter.

    Similar to the original car-following safety filter,
    but adapted for multi-lane state.
    """

    v_ego, lane_ego, distance_front, relative_velocity_front, front_gap_target, rear_gap_target = state

    safe_distance = 5.0 + 0.5 * v_ego

    if distance_front < safe_distance:
        return min(a_nom, -2.5)

    if distance_front < safe_distance + 5.0 and relative_velocity_front < 0:
        return min(a_nom, -1.0)

    return a_nom