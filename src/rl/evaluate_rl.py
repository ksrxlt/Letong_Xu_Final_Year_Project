import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env import CarFollowingEnv
from controllers import cbf_safety_filter


def action_to_acceleration(action):
    """
    Convert PPO action in [-1, 1] to physical acceleration in [-4, 2] m/s^2.
    """
    action_value = float(np.clip(action[0], -1.0, 1.0))
    acceleration = -4.0 + (action_value + 1.0) * 3.0
    return acceleration


def run_rl_episode(model, use_cbf=False, seed=0):
    """
    Run one episode using either:
    - RL only
    - RL + CBF safety filter
    """
    env = CarFollowingEnv()
    state = env.reset(seed=seed)

    logs = []
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        a_nom = action_to_acceleration(action)

        if use_cbf:
            a_ego = cbf_safety_filter(a_nom, state)
        else:
            a_ego = a_nom

        next_state, reward, done, info = env.step(a_ego)

        info["a_nom"] = a_nom
        info["use_cbf"] = use_cbf
        info["method"] = "RL + CBF" if use_cbf else "RL"
        info["reward"] = reward

        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    """
    Summarise one episode into evaluation metrics.
    """
    return {
        "collision": df["collision"].any(),
        "num_safety_violations": df["safety_violation"].sum(),
        "min_distance": df["distance"].min(),
        "mean_speed": df["v_ego"].mean(),
        "total_reward": df["reward"].sum(),
        "mean_abs_acceleration": df["a_ego"].abs().mean(),
    }


def evaluate_method(model, use_cbf=False, num_episodes=50):
    """
    Run multiple episodes and collect summary results.
    """
    summaries = []

    for seed in range(num_episodes):
        df = run_rl_episode(model, use_cbf=use_cbf, seed=seed)
        summary = summarize_episode(df)
        summary["seed"] = seed
        summary["method"] = "RL + CBF" if use_cbf else "RL"
        summaries.append(summary)

    return pd.DataFrame(summaries)


def main():
    model = PPO.load("ppo_car_following")

    num_episodes = 50

    rl_results = evaluate_method(model, use_cbf=False, num_episodes=num_episodes)
    rl_cbf_results = evaluate_method(model, use_cbf=True, num_episodes=num_episodes)

    all_results = pd.concat([rl_results, rl_cbf_results], ignore_index=True)

    summary_table = all_results.groupby("method").agg(
        collision_rate=("collision", "mean"),
        avg_safety_violations=("num_safety_violations", "mean"),
        min_distance_mean=("min_distance", "mean"),
        min_distance_worst=("min_distance", "min"),
        mean_speed=("mean_speed", "mean"),
        avg_total_reward=("total_reward", "mean"),
        mean_abs_acceleration=("mean_abs_acceleration", "mean"),
    )

    print("\nRL Evaluation Summary:")
    print(summary_table)

    all_results.to_csv("rl_all_episode_results.csv", index=False)
    summary_table.to_csv("rl_summary_table.csv")

    rl_example = run_rl_episode(model, use_cbf=False, seed=1)
    rl_cbf_example = run_rl_episode(model, use_cbf=True, seed=1)

    rl_example.to_csv("rl_result.csv", index=False)
    rl_cbf_example.to_csv("rl_cbf_result.csv", index=False)

    print("\nSaved:")
    print("rl_all_episode_results.csv")
    print("rl_summary_table.csv")
    print("rl_result.csv")
    print("rl_cbf_result.csv")


if __name__ == "__main__":
    main()