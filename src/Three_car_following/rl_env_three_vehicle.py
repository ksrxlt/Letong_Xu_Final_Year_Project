import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env_three_vehicle import ThreeVehicleFollowingEnv


class ThreeVehicleRLEnv(gym.Env):
    """
    Gymnasium wrapper for PPO training.

    PPO observes a normalized state.
    PPO action is normalized in [-1, 1].

    Action mapping:
        action = -1 -> acceleration = -4 m/s^2
        action =  0 -> acceleration =  0 m/s^2
        action =  1 -> acceleration = +2 m/s^2
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario="normal", target_distance=4000.0):
        super().__init__()

        self.env = ThreeVehicleFollowingEnv(
            scenario=scenario,
            target_distance=target_distance,
        )

        self.speed_limit = 130.0 / 3.6
        self.target_distance = target_distance

        sample_state = self.env.reset(seed=0)
        sample_obs = self.normalize_state(sample_state)
        self.state_dim = len(sample_obs)

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def normalize_state(self, state):
        """
        Original state:
        [
            v_ego,
            v_front_effective,
            distance_front,
            relative_velocity_front,
            cut_in_happened,
            remaining_distance
        ]

        PPO observation includes extra learning-friendly features:
            - normalized speeds
            - gap size
            - gap error
            - closing speed
            - remaining distance
        """

        v_ego = state[0]
        v_front = state[1]
        distance_front = state[2]
        relative_velocity = state[3]  # v_front - v_ego
        cut_in_happened = state[4]
        remaining_distance = state[5]

        closing_speed = v_ego - v_front

        # Same soft barrier logic as env
        soft_distance = 5.0 + 0.6 * v_ego + 1.0 * max(0.0, closing_speed)
        desired_distance = soft_distance + 3.0
        gap_error = distance_front - desired_distance

        normalized = np.array(
            [
                v_ego / self.speed_limit,
                v_front / self.speed_limit,

                # do not clip too aggressively
                np.clip(distance_front / 500.0, 0.0, 5.0),

                relative_velocity / self.speed_limit,
                closing_speed / self.speed_limit,

                np.clip(gap_error / 200.0, -5.0, 5.0),

                cut_in_happened,
                remaining_distance / self.target_distance,

                # explicit too-far indicator
                1.0 if gap_error > 30.0 else 0.0,
            ],
            dtype=np.float32,
        )

        return normalized

    def map_action_to_acceleration(self, action):
        """
        action = 0 must mean acceleration = 0.
        """

        action_value = float(np.clip(action[0], -1.0, 1.0))

        if action_value >= 0.0:
            acceleration = 2.0 * action_value
        else:
            acceleration = 4.0 * action_value

        return float(np.clip(acceleration, -4.0, 2.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self.env.reset(seed=seed)

        return self.normalize_state(state), {}

    def step(self, action):
        a_ego = self.map_action_to_acceleration(action)

        next_state, reward, done, info = self.env.step(a_ego)

        terminated = bool(
            info.get("collision", False)
            or info.get("reached_target", False)
        )

        truncated = bool(
            self.env.t >= self.env.max_steps
            and not terminated
        )

        return (
            self.normalize_state(next_state),
            float(reward),
            terminated,
            truncated,
            info,
        )