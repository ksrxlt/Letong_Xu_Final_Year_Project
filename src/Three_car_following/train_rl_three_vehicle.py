from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl_env_three_vehicle import ThreeVehicleRLEnv


def main():
    env = ThreeVehicleRLEnv(
        scenario="normal",
        target_distance=4000.0,
    )

    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
    )

    model.learn(total_timesteps=300_000)

    model.save("ppo_three_vehicle_cut_in")

    print("Saved model: ppo_three_vehicle_cut_in.zip")


if __name__ == "__main__":
    main()