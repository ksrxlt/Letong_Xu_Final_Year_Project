import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env_three_vehicle import ThreeVehicleFollowingEnv
from controllers_three_vehicle import cbf_three_vehicle_filter


def map_action_to_acceleration(action):
    action_value = float(np.clip(action[0], -1.0, 1.0))

    a_min = -4.0
    a_max = 2.0

    acceleration = a_min + (action_value + 1.0) * 0.5 * (a_max - a_min)

    return float(np.clip(acceleration, a_min, a_max))


def run_rl_episode(
    model,
    use_cbf=False,
    seed=0,
    scenario="normal",
    target_distance=4000.0,
):
    env = ThreeVehicleFollowingEnv(
        scenario=scenario,
        target_distance=target_distance,
    )

    state = env.reset(seed=seed)

    logs = []
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)

        # This is now PPO output, not rule-based output
        a_nom = map_action_to_acceleration(action)

        if use_cbf:
            a_ego = cbf_three_vehicle_filter(a_nom, state, dt=env.dt)
        else:
            a_ego = a_nom

        next_state, reward, done, info = env.step(a_ego)

        info["a_nom"] = a_nom
        info["a_ego"] = a_ego
        info["use_cbf"] = use_cbf
        info["method"] = "PPO + CBF" if use_cbf else "PPO without CBF"
        info["scenario"] = scenario

        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    if "safety_violation" in df.columns:
        safety_violations = df["safety_violation"].sum()
    elif "front_too_close" in df.columns:
        safety_violations = df["front_too_close"].sum()
    else:
        safety_violations = 0

    if "cut_in_happened" in df.columns:
        cut_in_happened = df["cut_in_happened"].any()
    else:
        cut_in_happened = False

    if "cut_in_left_this_step" in df.columns:
        cut_out_happened = df["cut_in_left_this_step"].any()
    else:
        cut_out_happened = False

    correction = (df["a_ego"] - df["a_nom"]).abs()

    return {
        "reached_target": df["reached_target"].any(),
        "travel_time": df["time"].iloc[-1],
        "collision": df["collision"].any(),
        "safety_violations": safety_violations,
        "cut_in_happened": cut_in_happened,
        "cut_out_happened": cut_out_happened,
        "min_front_distance": df["distance_front"].min(),
        "mean_front_distance": df["distance_front"].mean(),
        "mean_speed": df["v_ego"].mean(),
        "max_speed": df["v_ego"].max(),
        "total_reward": df["reward"].sum(),
        "mean_abs_acceleration": df["a_ego"].abs().mean(),
        "mean_cbf_correction": correction.mean(),
        "max_cbf_correction": correction.max(),
        "cbf_intervention_rate": (correction > 0.01).mean(),
    }


def evaluate_method(
    model,
    use_cbf=False,
    num_episodes=20,
    scenario="normal",
    target_distance=4000.0,
):
    summaries = []

    for seed in range(num_episodes):
        df = run_rl_episode(
            model=model,
            use_cbf=use_cbf,
            seed=seed,
            scenario=scenario,
            target_distance=target_distance,
        )

        summary = summarize_episode(df)
        summary["seed"] = seed
        summary["method"] = "PPO + CBF" if use_cbf else "PPO without CBF"
        summary["scenario"] = scenario

        summaries.append(summary)

    return pd.DataFrame(summaries)


def main():
    model = PPO.load("ppo_three_vehicle_cut_in")

    # Single example trajectory for plotting
    ppo_df = run_rl_episode(
        model,
        use_cbf=False,
        seed=1,
        scenario="normal",
        target_distance=4000.0,
    )

    ppo_cbf_df = run_rl_episode(
        model,
        use_cbf=True,
        seed=1,
        scenario="normal",
        target_distance=4000.0,
    )

    # Keep these names so plot_three_vehicle.py can directly use them
    ppo_df.to_csv("three_vehicle_baseline.csv", index=False)
    ppo_cbf_df.to_csv("three_vehicle_cbf.csv", index=False)

    print("PPO without CBF:")
    print(summarize_episode(ppo_df))

    print("\nPPO + CBF:")
    print(summarize_episode(ppo_cbf_df))

    # Multi-seed evaluation
    for scenario in ["normal", "hard"]:
        ppo_results = evaluate_method(
            model,
            use_cbf=False,
            num_episodes=20,
            scenario=scenario,
            target_distance=4000.0,
        )

        ppo_cbf_results = evaluate_method(
            model,
            use_cbf=True,
            num_episodes=20,
            scenario=scenario,
            target_distance=4000.0,
        )

        all_results = pd.concat(
            [ppo_results, ppo_cbf_results],
            ignore_index=True,
        )

        summary_table = all_results.groupby("method").agg(
            reached_target_rate=("reached_target", "mean"),
            mean_travel_time=("travel_time", "mean"),
            collision_rate=("collision", "mean"),
            avg_safety_violations=("safety_violations", "mean"),
            cut_in_rate=("cut_in_happened", "mean"),
            cut_out_rate=("cut_out_happened", "mean"),
            mean_min_front_distance=("min_front_distance", "mean"),
            worst_min_front_distance=("min_front_distance", "min"),
            mean_front_distance=("mean_front_distance", "mean"),
            mean_speed=("mean_speed", "mean"),
            max_speed=("max_speed", "mean"),
            avg_total_reward=("total_reward", "mean"),
            mean_abs_acceleration=("mean_abs_acceleration", "mean"),
            mean_cbf_correction=("mean_cbf_correction", "mean"),
            max_cbf_correction=("max_cbf_correction", "mean"),
            cbf_intervention_rate=("cbf_intervention_rate", "mean"),
        )

        all_results.to_csv(
            f"rl_three_vehicle_all_results_{scenario}.csv",
            index=False,
        )

        summary_table.to_csv(
            f"rl_three_vehicle_summary_{scenario}.csv"
        )

        print(f"\nScenario: {scenario}")
        print(summary_table)

    print("\nSaved RL evaluation results.")


if __name__ == "__main__":
    main()