from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_env import CarFollowingGymEnv


def main():
    env = CarFollowingGymEnv(scenario="normal")

    # Check whether the custom environment follows Gymnasium/SB3 API
    check_env(env, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_car_following")

    print("Saved RL model to ppo_car_following.zip")


if __name__ == "__main__":
    main()