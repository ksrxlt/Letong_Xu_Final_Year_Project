import numpy as np


class CarFollowingEnv:
    def __init__(self, dt=0.1, max_steps=300):
        self.dt = dt
        self.max_steps = max_steps
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.t = 0

        # Ego vehicle: autonomous vehicle
        self.x_ego = 0.0
        self.v_ego = 20.0

        # Front vehicle: human-driven vehicle
        self.x_front = 22.0
        self.v_front = 18.0

        self.done = False

        return self.get_state()

    def get_state(self):
        distance = self.x_front - self.x_ego
        relative_velocity = self.v_front - self.v_ego

        state = np.array(
            [
                self.v_ego,
                self.v_front,
                distance,
                relative_velocity,
            ],
            dtype=np.float32,
        )

        return state

    def human_driver_acceleration(self):
        """
        This represents uncertainty in human driving behaviour.
        Most of the time, the front car changes speed mildly.
        Sometimes, it brakes suddenly.
        """
        if np.random.rand() < 0.15:
            return np.random.uniform(-4.0, -2.0)  # sudden braking
        else:
            return np.random.uniform(-1.0, 1.0)   # normal behaviour

    def step(self, a_ego):
        # Limit ego vehicle acceleration
        a_ego = float(np.clip(a_ego, -4.0, 2.0))

        # Human-driven front vehicle acceleration
        a_front = self.human_driver_acceleration()

        # Update velocities
        self.v_ego = max(0.0, self.v_ego + a_ego * self.dt)
        self.v_front = max(0.0, self.v_front + a_front * self.dt)

        # Update positions
        self.x_ego = self.x_ego + self.v_ego * self.dt
        self.x_front = self.x_front + self.v_front * self.dt

        self.t += 1

        # Calculate distance
        distance = self.x_front - self.x_ego

        # Safety distance
        safe_distance = 5.0 + 0.5 * self.v_ego

        # Collision and safety violation checks
        collision = distance <= 2.0
        safety_violation = distance < safe_distance

        # Reward: this will be useful later for RL
        reward = self.v_ego * self.dt
        if collision:
            reward -= 100.0
        if safety_violation:
            reward -= 10.0
        reward -= 0.1 * (a_ego ** 2)

        self.done = collision or self.t >= self.max_steps

        next_state = self.get_state()

        info = {
            "step": self.t,
            "time": self.t * self.dt,
            "x_ego": self.x_ego,
            "v_ego": self.v_ego,
            "x_front": self.x_front,
            "v_front": self.v_front,
            "distance": distance,
            "safe_distance": safe_distance,
            "collision": collision,
            "safety_violation": safety_violation,
            "a_ego": a_ego,
            "a_front": a_front,
            "reward": reward,
        }

        return next_state, reward, self.done, info