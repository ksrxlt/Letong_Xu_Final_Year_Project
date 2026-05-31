import numpy as np


class ThreeVehicleFollowingEnv:
    """
    Cut-in car-following environment.

    Vehicles:
        ego vehicle:
            autonomous vehicle in lane 0

        front vehicle:
            original front vehicle in lane 0.
            It can suddenly accelerate or suddenly brake.

        cut-in vehicle:
            starts in the adjacent lane.
            If the original front vehicle accelerates away and the ego vehicle
            fails to keep up, creating a front gap larger than 20 m,
            the cut-in vehicle may suddenly enter lane 0 in front of ego.

    Goal:
        Ego vehicle should complete a fixed-distance driving task while
        maintaining safe distance to the effective front vehicle.
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

        # Speed limit: 130 km/h = 36.11 m/s
        self.speed_limit = 130.0 / 3.6

        # Cut-in is disabled during the initial warm-up stage.
        # 1000 m means cut-in can happen after about 50 s if ego drives around 20 m/s.
        self.cut_in_allowed_after_distance = 1000.0

        # Cut-in vehicle may leave after staying in ego lane for some time.
        self.cut_in_min_duration = 45.0
        self.cut_in_max_duration = 90.0

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

        # Original front vehicle in ego lane
        self.x_front = np.random.uniform(25.0, 35.0)
        self.v_front = np.random.uniform(17.0, 21.0)
        self.lane_front = 0

        # Cut-in vehicle initially in adjacent/right lane
        # It will only become relevant after it cuts in.
        self.x_cut_in = np.random.uniform(5.0, 25.0)
        self.v_cut_in = np.random.uniform(17.0, 22.0)
        self.lane_cut_in = 1

        self.cut_in_happened = False
        self.cut_in_active = False
        self.cut_in_time = None
        self.cut_in_leave_time = None

        # Cut-in vehicle event flags
        self.cut_in_event_type = "normal"   # "normal", "accelerate", "brake"
        self.cut_in_event_timer = 0

        # Front vehicle event flags
        self.front_event_type = "normal"     # "normal", "accelerate", "brake"
        self.front_event_timer = 0
        self.recent_front_acceleration_event = False

        return self.get_state()

    def safe_distance(self):
        """
        Minimum safe distance to the effective front vehicle.
        """
        return 5.0 + 0.5 * self.v_ego

    def get_effective_front_vehicle(self):
        """
        Return the effective front vehicle.

        Before cut-in:
            Follow the original front vehicle.

        While cut-in vehicle is active:
            Follow the cut-in vehicle.

        After cut-in vehicle leaves:
            Follow the original front vehicle again.
        """

        # If cut-in vehicle is currently in ego lane, it becomes the effective front vehicle
        if self.cut_in_active and self.lane_cut_in == self.lane_ego and self.x_cut_in > self.x_ego:
            return ("cut_in", self.x_cut_in, self.v_cut_in)

        # Otherwise follow the original front vehicle
        if self.x_front > self.x_ego:
            return ("front", self.x_front, self.v_front)

        # If no vehicle ahead, treat as open road
        return ("none", self.x_ego + 999.0, self.v_ego)

    def get_state(self):
        front_type, x_front_effective, v_front_effective = self.get_effective_front_vehicle()

        distance_front = x_front_effective - self.x_ego
        relative_velocity_front = v_front_effective - self.v_ego
        remaining_distance = max(0.0, self.target_distance - self.x_ego)

        return np.array(
            [
                self.v_ego,
                v_front_effective,
                distance_front,
                relative_velocity_front,
                float(self.cut_in_happened),
                remaining_distance,
            ],
            dtype=np.float32,
        )

    def update_front_event(self):
        """
        Decide whether the original front vehicle suddenly accelerates
        or suddenly brakes.

        The event lasts for several simulation steps.
        """

        self.recent_front_acceleration_event = False

        # If an event is already active, continue it.
        if self.front_event_timer > 0:
            self.front_event_timer -= 1

            if self.front_event_type == "accelerate":
                self.recent_front_acceleration_event = True

            if self.front_event_timer == 0:
                self.front_event_type = "normal"

            return

        # Start a new event.
        if self.scenario == "normal":
            accel_prob = 0.003
            brake_prob = 0.003

        elif self.scenario == "hard":
            accel_prob = 0.006
            brake_prob = 0.006

        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        r = np.random.rand()

        if r < accel_prob:
            self.front_event_type = "accelerate"
            self.front_event_timer = np.random.randint(15, 35)  # 1.5–3.5 seconds
            self.recent_front_acceleration_event = True

        elif r < accel_prob + brake_prob:
            self.front_event_type = "brake"
            self.front_event_timer = np.random.randint(15, 35)  # 1.5–3.5 seconds

        else:
            self.front_event_type = "normal"

    def front_vehicle_acceleration(self):
        """
        Original front vehicle behaviour.

        It normally follows a target speed, but can sometimes suddenly
        accelerate or suddenly brake.
        """

        self.update_front_event()

        target_speed = 20.0

        # Normal mean-reverting behaviour
        a_front = 0.35 * (target_speed - self.v_front)
        a_front += np.random.uniform(-0.5, 0.5)

        if self.front_event_type == "accelerate":
            # Sudden acceleration event
            a_front += np.random.uniform(2.5, 4.0)

        elif self.front_event_type == "brake":
            # Sudden braking event
            a_front += np.random.uniform(-5.0, -3.0)

        return float(np.clip(a_front, -5.0, 3.0))

    def cut_in_vehicle_acceleration(self):
        """
        Cut-in vehicle behaviour.

        Before cut-in:
            It drives normally in the adjacent lane.

        During active cut-in:
            It may suddenly accelerate or suddenly brake.

        After leaving:
            It continues driving in adjacent lane, but no longer affects ego.
        """

        target_speed = 19.0

        a_cut = 0.35 * (target_speed - self.v_cut_in)
        a_cut += np.random.uniform(-0.7, 0.7)

        # Only create aggressive events while it is actively in ego lane
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
                accel_prob = 0.006
                brake_prob = 0.006

            elif self.scenario == "hard":
                accel_prob = 0.012
                brake_prob = 0.012

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
        Cut-in condition:

        Cut-in is disabled during the early warm-up stage.

        After that, if the ego vehicle leaves a gap larger than 20 m to the
        original front vehicle, a neighbouring vehicle cuts into the ego lane.

        The cut-in position is random and can be very close to ego.
        """

        if self.cut_in_active:
            return False

        # Only allow one cut-in event in this episode
        if self.cut_in_happened:
            return False

        # Do not allow cut-in too early
        if self.x_ego < self.cut_in_allowed_after_distance:
            return False

        distance_to_original_front = self.x_front - self.x_ego

        # Main trigger: ego leaves a large gap
        gap_too_large = distance_to_original_front > 20.0

        if gap_too_large:
            r = np.random.rand()

            if r < 0.35:
                # Aggressive cut-in, very close
                cut_in_gap = np.random.uniform(3.0, 8.0)
            elif r < 0.75:
                # Moderate cut-in
                cut_in_gap = np.random.uniform(8.0, 14.0)
            else:
                # Safer cut-in
                cut_in_gap = np.random.uniform(14.0, 20.0)

            self.x_cut_in = self.x_ego + cut_in_gap

            # Cut-in vehicle may be slower or faster than ego
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

        return False
    
    def maybe_cut_in_leave(self):
        """
        The cut-in vehicle may leave ego lane after some time.

        If it leaves, the ego vehicle will again consider the original front
        vehicle as the effective front vehicle. This can create a large open gap,
        so ego should accelerate to catch up, but must respect the speed limit.
        """

        if not self.cut_in_active:
            return False

        current_time = self.t * self.dt
        time_since_cut_in = current_time - self.cut_in_time

        if time_since_cut_in < self.cut_in_min_duration:
            return False

        gap_to_cut_in = self.x_cut_in - self.x_ego

        # If cut-in vehicle has moved sufficiently ahead, it may leave.
        leave_due_to_gap = gap_to_cut_in > 28.0 and np.random.rand() < 0.01

        # If it has stayed for a long time, it may also leave.
        leave_due_to_time = time_since_cut_in > self.cut_in_max_duration and np.random.rand() < 0.08

        if leave_due_to_gap or leave_due_to_time:
            self.lane_cut_in = 1
            self.cut_in_active = False
            self.cut_in_leave_time = current_time
            return True

        return False

    def step(self, a_ego):
        a_ego = float(np.clip(a_ego, -4.0, 2.0))

        # Vehicle accelerations
        a_front = self.front_vehicle_acceleration()
        a_cut_in = self.cut_in_vehicle_acceleration()

        # Cut-in may happen after front acceleration event is generated
        cut_in_this_step = self.maybe_cut_in()

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

        # Cut-in vehicle may leave ego lane after moving for some time
        cut_in_left_this_step = self.maybe_cut_in_leave()

        self.t += 1

        front_type, x_front_effective, v_front_effective = self.get_effective_front_vehicle()

        distance_front = x_front_effective - self.x_ego
        safe_distance = self.safe_distance()

        collision = distance_front <= 2.0
        safety_violation = distance_front < safe_distance

        # Too far is not collision risk, but it means ego is inefficient / not following.
        too_far = distance_front > safe_distance + 25.0

        reached_target = self.x_ego >= self.target_distance

        # Reward
        reward = self.v_ego * self.dt

        if reached_target:
            reward += 500.0

        if collision:
            reward -= 1000.0

        if safety_violation:
            reward -= 10.0

        if too_far:
            reward -= 1.0

        if cut_in_this_step:
            # Cut-in is an external disturbance.
            reward -= 2.0

        # Smoothness / control effort penalty
        reward -= 0.1 * (a_ego ** 2)

        self.done = collision or reached_target or self.t >= self.max_steps

        next_state = self.get_state()

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
            "distance_front": distance_front,
            "safe_distance": safe_distance,
            "too_far": too_far,

            "front_event_type": self.front_event_type,
            "front_event_timer": self.front_event_timer,

            "cut_in_happened": self.cut_in_happened,
            "cut_in_active": self.cut_in_active,
            "cut_in_this_step": cut_in_this_step,
            "cut_in_left_this_step": cut_in_left_this_step,
            "cut_in_time": self.cut_in_time,
            "cut_in_leave_time": self.cut_in_leave_time,
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