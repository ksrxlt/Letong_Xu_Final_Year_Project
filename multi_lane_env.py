import numpy as np


class MultiLaneCarFollowingEnv:
    """
    Simplified two-lane car-following and lane-change environment.

    Lanes:
        lane 0: original lane
        lane 1: adjacent/overtaking lane

    Vehicles:
        ego vehicle: autonomous vehicle
        slow vehicle: front vehicle in lane 0
        side vehicle: vehicle in lane 1

    Simplification:
        - Vehicles move only in the longitudinal x direction.
        - Lane change is represented by changing lane index.
        - No lateral dynamics are modelled.
    """

    def __init__(self, dt=0.1, max_steps=300, scenario="safe_overtake"):
        self.dt = dt
        self.max_steps = max_steps
        self.scenario = scenario
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.done = False

        # Ego vehicle
        self.x_ego = 0.0
        self.v_ego = 20.0
        self.lane_ego = 0

        # Slow vehicle in the original lane
        self.x_slow = 30.0
        self.v_slow = 12.0
        self.lane_slow = 0

        # Side-lane vehicle
        self.lane_side = 1

        if self.scenario == "safe_overtake":
            # Side vehicle is far enough, lane change should be allowed
            self.x_side = 70.0
            self.v_side = 18.0

        elif self.scenario == "blocked_by_front":
            # Side vehicle is ahead but too close, lane change should be blocked
            self.x_side = 18.0
            self.v_side = 16.0

        elif self.scenario == "blocked_by_rear":
            # Side vehicle is behind but too close, lane change should be blocked
            self.x_side = -8.0
            self.v_side = 24.0

        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        return self.get_state()

    def get_vehicles(self):
        return [
            {
                "name": "slow",
                "x": self.x_slow,
                "v": self.v_slow,
                "lane": self.lane_slow,
            },
            {
                "name": "side",
                "x": self.x_side,
                "v": self.v_side,
                "lane": self.lane_side,
            },
        ]

    def find_front_vehicle_same_lane(self):
        vehicles = self.get_vehicles()

        front_vehicles = [
            veh for veh in vehicles
            if veh["lane"] == self.lane_ego and veh["x"] > self.x_ego
        ]

        if len(front_vehicles) == 0:
            return None

        return min(front_vehicles, key=lambda veh: veh["x"] - self.x_ego)

    def get_target_lane_gaps(self):
        target_lane = 1 - self.lane_ego
        vehicles = self.get_vehicles()

        front_gaps = []
        rear_gaps = []

        for veh in vehicles:
            if veh["lane"] != target_lane:
                continue

            gap = veh["x"] - self.x_ego

            if gap >= 0:
                front_gaps.append(gap)
            else:
                rear_gaps.append(-gap)

        if len(front_gaps) == 0:
            front_gap = 999.0
        else:
            front_gap = min(front_gaps)

        if len(rear_gaps) == 0:
            rear_gap = 999.0
        else:
            rear_gap = min(rear_gaps)

        return front_gap, rear_gap

    def get_state(self):
        front_vehicle = self.find_front_vehicle_same_lane()

        if front_vehicle is None:
            distance_front = 999.0
            relative_velocity_front = 0.0
        else:
            distance_front = front_vehicle["x"] - self.x_ego
            relative_velocity_front = front_vehicle["v"] - self.v_ego

        front_gap_target, rear_gap_target = self.get_target_lane_gaps()

        state = np.array(
            [
                self.v_ego,
                float(self.lane_ego),
                distance_front,
                relative_velocity_front,
                front_gap_target,
                rear_gap_target,
            ],
            dtype=np.float32,
        )

        return state

    def human_vehicle_acceleration(self):
        """
        Mild stochastic acceleration for non-ego vehicles.
        """
        return np.random.uniform(-0.5, 0.5)

    def safe_distance(self):
        return 5.0 + 0.5 * self.v_ego

    def step(self, a_ego, lane_change_command=0):
        """
        Args:
            a_ego: ego longitudinal acceleration
            lane_change_command:
                0 = keep lane
                1 = request lane change to adjacent lane
        """

        a_ego = float(np.clip(a_ego, -4.0, 2.0))

        # Lane change decision
        lane_change_executed = False
        unsafe_lane_change_attempt = False

        if lane_change_command == 1:
            if self.is_target_lane_safe():
                self.lane_ego = 1 - self.lane_ego
                lane_change_executed = True
            else:
                unsafe_lane_change_attempt = True

        # Human-driven vehicles update
        a_slow = self.human_vehicle_acceleration()
        a_side = self.human_vehicle_acceleration()

        self.v_ego = max(0.0, self.v_ego + a_ego * self.dt)
        self.v_slow = max(0.0, self.v_slow + a_slow * self.dt)
        self.v_side = max(0.0, self.v_side + a_side * self.dt)

        self.x_ego += self.v_ego * self.dt
        self.x_slow += self.v_slow * self.dt
        self.x_side += self.v_side * self.dt

        self.t += 1

        # Safety checks
        collision = self.check_collision()
        safety_violation = self.check_safety_violation()

        reward = self.v_ego * self.dt
        reward -= 100.0 if collision else 0.0
        reward -= 10.0 if safety_violation else 0.0
        reward -= 5.0 if unsafe_lane_change_attempt else 0.0
        reward -= 0.1 * (a_ego ** 2)

        self.done = collision or self.t >= self.max_steps

        next_state = self.get_state()

        front_vehicle = self.find_front_vehicle_same_lane()
        if front_vehicle is None:
            distance_front = 999.0
        else:
            distance_front = front_vehicle["x"] - self.x_ego

        front_gap_target, rear_gap_target = self.get_target_lane_gaps()

        info = {
            "step": self.t,
            "time": self.t * self.dt,
            "x_ego": self.x_ego,
            "v_ego": self.v_ego,
            "lane_ego": self.lane_ego,
            "x_slow": self.x_slow,
            "v_slow": self.v_slow,
            "lane_slow": self.lane_slow,
            "x_side": self.x_side,
            "v_side": self.v_side,
            "lane_side": self.lane_side,
            "distance_front": distance_front,
            "safe_distance": self.safe_distance(),
            "front_gap_target": front_gap_target,
            "rear_gap_target": rear_gap_target,
            "collision": collision,
            "safety_violation": safety_violation,
            "unsafe_lane_change_attempt": unsafe_lane_change_attempt,
            "lane_change_executed": lane_change_executed,
            "a_ego": a_ego,
            "reward": reward,
        }

        return next_state, reward, self.done, info

    def is_target_lane_safe(self):
        front_gap, rear_gap = self.get_target_lane_gaps()

        front_safe_distance = 15.0
        rear_safe_distance = 10.0

        return front_gap > front_safe_distance and rear_gap > rear_safe_distance

    def check_collision(self):
        vehicles = self.get_vehicles()

        for veh in vehicles:
            if veh["lane"] == self.lane_ego:
                distance = abs(veh["x"] - self.x_ego)
                if distance <= 2.0:
                    return True

        return False

    def check_safety_violation(self):
        front_vehicle = self.find_front_vehicle_same_lane()

        if front_vehicle is None:
            return False

        distance_front = front_vehicle["x"] - self.x_ego
        return distance_front < self.safe_distance()