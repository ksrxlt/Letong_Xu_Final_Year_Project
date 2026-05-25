import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import CarFollowingEnv


class CarFollowingGymEnv(gym.Env):
    """
    Gymnasium wrapper for the car-following environment.

    Observation:
        [v_ego, v_front, distance, relative_velocity]

    Action:
        Normalised acceleration command in [-1, 1].
        It is mapped to physical acceleration in [-4, 2] m/s^2.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario="normal"):
        super().__init__()

        self.scenario = scenario
        self.env = CarFollowingEnv()

        # Observation space:
        # v_ego, v_front, distance, relative_velocity
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -100.0, -50.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 200.0, 50.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Normalised action space.
        # Stable-Baselines3 recommends normalising continuous actions.
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self.env.reset(seed=seed)

        return state.astype(np.float32), {}

    def step(self, action):
        # Convert action from [-1, 1] to acceleration [-4, 2]
        action_value = float(np.clip(action[0], -1.0, 1.0))
        a_ego = -4.0 + (action_value + 1.0) * 3.0

        next_state, reward, done, info = self.env.step(a_ego)

        terminated = bool(info["collision"])
        truncated = bool(self.env.t >= self.env.max_steps)

        return next_state.astype(np.float32), float(reward), terminated, truncated, info