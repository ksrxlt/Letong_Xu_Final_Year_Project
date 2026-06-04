import numpy as np


class ThreeVehicleFollowingEnv:
    """
    Cut-in / cut-out car-following environment.

    Main control structure:
        PPO without CBF:
            state -> PPO -> a_nom -> ego

        PPO + CBF:
            state -> PPO -> a_nom -> CBF -> a_ego -> ego

    Vehicles:
        ego:
            controlled vehicle.

        vehicle 1:
            original front vehicle.

        vehicle 2:
            cut-in vehicle. It starts in the adjacent lane. If ego leaves
            a large enough gap, vehicle 2 cuts in. Later it may leave, and
            ego should catch up with vehicle 1 again.

    State:
        [
            v_ego,
            v_front_effective,
            distance_front,
            relative_velocity_front,
            cut_in_happened,
            remaining_distance
        ]
    """

    def __init__(
        self,
        dt=0.1,
        max_steps=3000,
        target_distance=4000.0,
        scenario="normal",
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.target_distance = target_distance
        self.scenario = scenario

        # 130 km/h = 36.11 m/s
        self.speed_limit = 130.0 / 3.6

        # Cut-in is disabled during the initial warm-up phase.
        self.cut_in_allowed_after_distance = 1000.0

        # Cut-in vehicle stays in ego lane for a while, then may leave.
        self.cut_in_min_duration = 45.0
        self.cut_in_max_duration = 90.0

        # ------------------------------------------------------------
        # Recovery gap after cut-out
        # ------------------------------------------------------------
        # When vehicle 2 leaves, vehicle 1 will be placed ahead of ego
        # with a controlled gap. This avoids the front_type="none" /
        # distance_front=999 issue and creates a clear catch-up task.
        self.recovery_gap_min = 80.0
        self.recovery_gap_max = 350.0

        # Sometimes vehicle 1 is placed much farther ahead, forcing ego
        # to accelerate close to the speed limit during recovery.
        self.recovery_gap_extreme_prob = 0.25
        self.recovery_gap_extreme_min = 350.0
        self.recovery_gap_extreme_max = 600.0

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.done = False

        # Ego vehicle
        self.x_ego = 0.0
        self.v_ego = np.random.uniform(18.0, 22.0)
        self.lane_ego = 0

        # Vehicle 1: original front vehicle
        self.x_front = np.random.uniform(25.0, 35.0)
        self.v_front = np.random.uniform(17.0, 21.0)
        self.lane_front = 0

        # Vehicle 2: cut-in vehicle, initially in adjacent lane
        self.x_cut_in = np.random.uniform(5.0, 25.0)
        self.v_cut_in = np.random.uniform(17.0, 22.0)
        self.lane_cut_in = 1

        # Cut-in state
        self.cut_in_happened = False
        self.cut_in_active = False
        self.cut_in_time = None
        self.cut_in_leave_time = None

        # Vehicle 1 event state
        self.front_event_type = "normal"
        self.front_event_timer = 0

        # Vehicle 2 event state
        self.cut_in_event_type = "normal"
        self.cut_in_event_timer = 0

        return self.get_state()

    # ============================================================
    # Soft / hard barriers
    # ============================================================

    def soft_barrier_distance(self, v_front_eff):
        """
        Soft comfort/reference barrier.

        Assumption:
            The front vehicle continues normally and does not suddenly
            emergency brake.

        This is used for reward shaping and plotting. Ego is allowed to go
        below this line temporarily, but it will receive a penalty.
        """

        min_gap = 5.0
        closing_speed = max(0.0, self.v_ego - v_front_eff)

        soft_distance = (
            min_gap
            + 0.6 * self.v_ego
            + 1.0 * closing_speed
        )

        return soft_distance

    def hard_barrier_distance(self, v_front_eff):
        """
        Hard short-horizon safety barrier.

        This is the minimum safety boundary used by the CBF.
        It is stricter than the previous version but still less conservative
        than the soft comfort barrier.
        """

        min_gap = 4.0
        closing_speed = max(0.0, self.v_ego - v_front_eff)

        hard_distance = (
            min_gap
            + 0.35 * self.v_ego
            + 0.80 * closing_speed
        )

        return hard_distance

    def safe_distance(self):
        """
        Kept for compatibility with old plotting code.

        Here, safe_distance means the soft comfort/reference barrier.
        """

        _, _, v_front_eff = self.get_effective_front_vehicle()
        return self.soft_barrier_distance(v_front_eff)

    # ============================================================
    # Vehicle selection and state
    # ============================================================

    def get_effective_front_vehicle(self):
        """
        Before cut-in:
            effective front vehicle = vehicle 1.

        During cut-in:
            effective front vehicle = vehicle 2.

        After cut-out:
            effective front vehicle = vehicle 1 again.

        If vehicle 1 is not ahead of ego, return "none". However, the
        cut-out recovery logic should normally prevent this by forcing
        vehicle 1 to be ahead of ego at the cut-out moment.
        """

        if (
            self.cut_in_active
            and self.lane_cut_in == self.lane_ego
            and self.x_cut_in > self.x_ego
        ):
            return "cut_in", self.x_cut_in, self.v_cut_in

        if self.x_front > self.x_ego:
            return "front", self.x_front, self.v_front

        return "none", self.x_ego + 999.0, self.v_ego

    def get_state(self):
        _, x_front_eff, v_front_eff = self.get_effective_front_vehicle()

        distance_front = x_front_eff - self.x_ego
        relative_velocity_front = v_front_eff - self.v_ego
        remaining_distance = max(0.0, self.target_distance - self.x_ego)

        return np.array(
            [
                self.v_ego,
                v_front_eff,
                distance_front,
                relative_velocity_front,
                float(self.cut_in_happened),
                remaining_distance,
            ],
            dtype=np.float32,
        )

    # ============================================================
    # Vehicle 1 behaviour
    # ============================================================

    def update_front_event(self):
        """
        Vehicle 1 may suddenly accelerate or brake.
        """

        if self.front_event_timer > 0:
            self.front_event_timer -= 1

            if self.front_event_timer == 0:
                self.front_event_type = "normal"

            return

        if self.scenario == "normal":
            accel_prob = 0.004
            brake_prob = 0.004
        elif self.scenario == "hard":
            accel_prob = 0.008
            brake_prob = 0.008
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        r = np.random.rand()

        if r < accel_prob:
            self.front_event_type = "accelerate"
            self.front_event_timer = np.random.randint(15, 35)
        elif r < accel_prob + brake_prob:
            self.front_event_type = "brake"
            self.front_event_timer = np.random.randint(15, 35)
        else:
            self.front_event_type = "normal"

    def front_vehicle_acceleration(self):
        """
        Vehicle 1 behaviour.
        """

        self.update_front_event()

        target_speed = 20.0

        a_front = 0.35 * (target_speed - self.v_front)
        a_front += np.random.uniform(-0.5, 0.5)

        if self.front_event_type == "accelerate":
            a_front += np.random.uniform(2.5, 4.0)

        elif self.front_event_type == "brake":
            a_front += np.random.uniform(-5.0, -3.0)

        return float(np.clip(a_front, -5.0, 3.0))

    # ============================================================
    # Vehicle 2 behaviour
    # ============================================================

    def cut_in_vehicle_acceleration(self):
        """
        Vehicle 2 behaviour.

        Before cut-in:
            normal adjacent-lane driving.

        During cut-in:
            may suddenly accelerate or brake.

        After cut-out:
            continues in adjacent lane and no longer affects ego.
        """

        target_speed = 19.0

        a_cut = 0.35 * (target_speed - self.v_cut_in)
        a_cut += np.random.uniform(-0.7, 0.7)

        if not self.cut_in_active:
            return float(np.clip(a_cut, -3.0, 2.5))

        if self.cut_in_event_timer > 0:
            self.cut_in_event_timer -= 1

            if self.cut_in_event_type == "accelerate":
                a_cut += np.random.uniform(2.5, 4.0)

            elif self.cut_in_event_type == "brake":
                a_cut += np.random.uniform(-5.0, -3.0)

            if self.cut_in_event_timer == 0:
                self.cut_in_event_type = "normal"

        else:
            if self.scenario == "normal":
                accel_prob = 0.008
                brake_prob = 0.008
            elif self.scenario == "hard":
                accel_prob = 0.016
                brake_prob = 0.016
            else:
                raise ValueError(f"Unknown scenario: {self.scenario}")

            r = np.random.rand()

            if r < accel_prob:
                self.cut_in_event_type = "accelerate"
                self.cut_in_event_timer = np.random.randint(15, 35)
                a_cut += np.random.uniform(2.5, 4.0)

            elif r < accel_prob + brake_prob:
                self.cut_in_event_type = "brake"
                self.cut_in_event_timer = np.random.randint(15, 35)
                a_cut += np.random.uniform(-5.0, -3.0)

            else:
                self.cut_in_event_type = "normal"

        return float(np.clip(a_cut, -5.0, 3.0))

    def maybe_cut_in(self):
        """
        Vehicle 2 cuts in if ego leaves a gap larger than 20 m after warm-up.
        """

        if self.cut_in_happened or self.cut_in_active:
            return False

        if self.x_ego < self.cut_in_allowed_after_distance:
            return False

        distance_to_vehicle_1 = self.x_front - self.x_ego

        if distance_to_vehicle_1 <= 20.0:
            return False

        r = np.random.rand()

        if r < 0.35:
            cut_in_gap = np.random.uniform(3.0, 8.0)
        elif r < 0.75:
            cut_in_gap = np.random.uniform(8.0, 14.0)
        else:
            cut_in_gap = np.random.uniform(14.0, 20.0)

        self.x_cut_in = self.x_ego + cut_in_gap

        self.v_cut_in = np.random.uniform(
            max(5.0, self.v_ego - 6.0),
            min(28.0, self.v_ego + 4.0),
        )

        self.lane_cut_in = 0
        self.cut_in_happened = True
        self.cut_in_active = True
        self.cut_in_time = self.t * self.dt

        self.cut_in_event_type = "normal"
        self.cut_in_event_timer = 0

        return True

    # ============================================================
    # Cut-out and recovery scenario
    # ============================================================

    def sample_recovery_gap_after_cut_out(self):
        """
        Sample the distance between ego and vehicle 1 after vehicle 2 leaves.

        This creates a controlled recovery scenario:
            - sometimes vehicle 1 is moderately far ahead
            - sometimes vehicle 1 is very far ahead, so ego must accelerate hard

        This prevents the case where ego has already passed vehicle 1 and
        front_type becomes "none" after cut-out.
        """

        if np.random.rand() < self.recovery_gap_extreme_prob:
            return np.random.uniform(
                self.recovery_gap_extreme_min,
                self.recovery_gap_extreme_max,
            )

        return np.random.uniform(
            self.recovery_gap_min,
            self.recovery_gap_max,
        )

    def maybe_cut_in_leave(self):
        """
        Vehicle 2 may leave ego lane after staying for a while.

        When it leaves, vehicle 1 is deliberately placed ahead of ego with
        a controlled recovery gap. This creates a clean cut-out recovery
        task and avoids the 999 m "no front vehicle" artefact.
        """

        if not self.cut_in_active:
            return False

        current_time = self.t * self.dt
        time_since_cut_in = current_time - self.cut_in_time

        if time_since_cut_in < self.cut_in_min_duration:
            return False

        gap_to_cut_in = self.x_cut_in - self.x_ego

        leave_due_to_gap = gap_to_cut_in > 40.0 and np.random.rand() < 0.01
        leave_due_to_time = (
            time_since_cut_in > self.cut_in_max_duration
            and np.random.rand() < 0.08
        )

        if leave_due_to_gap or leave_due_to_time:
            self.lane_cut_in = 1
            self.cut_in_active = False
            self.cut_in_leave_time = current_time

            # ------------------------------------------------------------
            # Controlled recovery setup:
            # Vehicle 1 is guaranteed to be ahead of ego after cut-out.
            # ------------------------------------------------------------
            recovery_gap = self.sample_recovery_gap_after_cut_out()
            self.x_front = self.x_ego + recovery_gap

            # Keep vehicle 1 at a realistic traffic speed.
            self.v_front = np.random.uniform(18.0, 22.0)

            # Reset vehicle 1 event state to avoid an immediate unnatural jump.
            self.front_event_type = "normal"
            self.front_event_timer = 0

            return True

        return False

    # ============================================================
    # Main step
    # ============================================================

    def step(self, a_ego):
        a_ego = float(np.clip(a_ego, -4.0, 2.0))

        # Cut-in can happen before motion update.
        cut_in_this_step = self.maybe_cut_in()

        a_front = self.front_vehicle_acceleration()
        a_cut_in = self.cut_in_vehicle_acceleration()

        # Update velocities
        self.v_ego = max(0.0, self.v_ego + a_ego * self.dt)
        self.v_ego = min(self.v_ego, self.speed_limit)

        self.v_front = max(0.0, self.v_front + a_front * self.dt)
        self.v_front = min(self.v_front, self.speed_limit)

        self.v_cut_in = max(0.0, self.v_cut_in + a_cut_in * self.dt)
        self.v_cut_in = min(self.v_cut_in, self.speed_limit)

        # Update positions
        self.x_ego += self.v_ego * self.dt
        self.x_front += self.v_front * self.dt
        self.x_cut_in += self.v_cut_in * self.dt

        self.t += 1

        cut_in_left_this_step = self.maybe_cut_in_leave()

        front_type, x_front_eff, v_front_eff = self.get_effective_front_vehicle()

        distance_front = x_front_eff - self.x_ego

        soft_distance = self.soft_barrier_distance(v_front_eff)
        hard_distance = self.hard_barrier_distance(v_front_eff)

        # Kept for plot compatibility.
        safe_distance = soft_distance

        collision = distance_front <= 2.0

        # Real safety violation uses hard barrier.
        safety_violation = distance_front < hard_distance

        # Too-far condition uses soft comfort barrier.
        too_far = distance_front > soft_distance + 25.0

        reached_target = self.x_ego >= self.target_distance

        # ============================================================
        # Stable PPO reward: learn basic acceleration and braking
        # ============================================================

        reward = 0.0

        desired_distance = soft_distance + 3.0
        distance_error = distance_front - desired_distance
        gap_excess = max(0.0, distance_error)
        too_close = max(0.0, -distance_error)

        closing_speed = self.v_ego - v_front_eff

        # 1. Progress reward
        reward += 0.1 * self.v_ego

        # 2. Distance shaping, bounded
        # Too far: penalise, but bounded.
        reward -= 4.0 * np.tanh(gap_excess / 100.0)

        # Too close: stronger penalty.
        reward -= 6.0 * np.tanh(too_close / 20.0)

        # 3. Action guidance when too far
        # If gap is large, encourage positive acceleration.
        if gap_excess > 20.0:
            reward += 2.0 * np.tanh(a_ego / 1.0)

            if a_ego < 0.0:
                reward -= 2.0 * abs(a_ego)

        # 4. Action guidance when too close
        # If too close, encourage braking.
        if distance_front < soft_distance:
            reward += 2.0 * np.tanh((-a_ego) / 2.0)

            if a_ego > 0.0:
                reward -= 3.0 * a_ego

        # 5. Closing speed guidance
        # If too far, ego should be faster than front vehicle.
        if gap_excess > 20.0:
            reward += 2.0 * np.tanh(closing_speed / 5.0)

        # If too close and ego is still faster, penalise.
        if distance_front < soft_distance and closing_speed > 0.0:
            reward -= 4.0 * np.tanh(closing_speed / 5.0)

        # 6. Hard barrier penalty
        if distance_front < hard_distance:
            reward -= 30.0 * np.tanh((hard_distance - distance_front) / 5.0)

        # 7. Collision penalty
        if collision:
            reward -= 100.0

        # 8. Target reward
        if reached_target:
            reward += 100.0

        # 9. Low speed penalty
        if self.v_ego < 5.0 and not reached_target:
            reward -= 5.0

        # 10. Speed limit penalty
        if self.v_ego > self.speed_limit:
            reward -= 20.0 * (self.v_ego - self.speed_limit)

        # 11. Small action penalty
        reward -= 0.001 * (a_ego ** 2)

        self.done = collision or reached_target or self.t >= self.max_steps

        next_state = self.get_state()

        gap_to_v1 = self.x_front - self.x_ego
        gap_to_v2 = self.x_cut_in - self.x_ego

        if front_type == "front":
            front_id = 1
        elif front_type == "cut_in":
            front_id = 2
        else:
            front_id = -1

        info = {
            "step": self.t,
            "time": self.t * self.dt,

            "x_ego": self.x_ego,
            "v_ego": self.v_ego,

            "x_front": self.x_front,
            "v_front": self.v_front,
            "lane_front": self.lane_front,

            "x_cut_in": self.x_cut_in,
            "v_cut_in": self.v_cut_in,
            "lane_cut_in": self.lane_cut_in,

            "front_type": front_type,
            "front_id": front_id,
            "distance_front": distance_front,
            "gap_to_v1": gap_to_v1,
            "gap_to_v2": gap_to_v2,

            # For old and new plots
            "safe_distance": safe_distance,
            "soft_barrier_distance": soft_distance,
            "hard_barrier_distance": hard_distance,

            "too_far": too_far,

            "cut_in_happened": self.cut_in_happened,
            "cut_in_active": self.cut_in_active,
            "cut_in_this_step": cut_in_this_step,
            "cut_in_left_this_step": cut_in_left_this_step,
            "cut_in_time": self.cut_in_time,
            "cut_in_leave_time": self.cut_in_leave_time,

            "front_event_type": self.front_event_type,
            "front_event_timer": self.front_event_timer,
            "cut_in_event_type": self.cut_in_event_type,
            "cut_in_event_timer": self.cut_in_event_timer,

            "collision": collision,
            "safety_violation": safety_violation,
            "reached_target": reached_target,

            "a_ego": a_ego,
            "a_front": a_front,
            "a_cut_in": a_cut_in,

            "reward": reward,
        }

        return next_state, reward, self.done, info