import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env_three_vehicle import ThreeVehicleFollowingEnv


class ThreeVehicleRLEnv(gym.Env):
    """
    Gymnasium wrapper for PPO training.

    PPO action:
        normalized action in [-1, 1]

    Physical acceleration:
        [-4, 2] m/s^2
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario="normal", target_distance=4000.0):
        super().__init__()

        self.env = ThreeVehicleFollowingEnv(
            scenario=scenario,
            target_distance=target_distance,
        )

        sample_state = self.env.reset(seed=0)
        self.state_dim = len(sample_state)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def map_action_to_acceleration(self, action):
        action_value = float(np.clip(action[0], -1.0, 1.0))

        a_min = -4.0
        a_max = 2.0

        acceleration = a_min + (action_value + 1.0) * 0.5 * (a_max - a_min)

        return float(np.clip(acceleration, a_min, a_max))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self.env.reset(seed=seed)

        return state.astype(np.float32), {}

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
            next_state.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )