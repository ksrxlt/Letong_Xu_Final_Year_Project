import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env_three_vehicle import ThreeVehicleFollowingEnv
from controllers_three_vehicle import cbf_three_vehicle_filter


def normalize_state(state, env, target_distance=4000.0):
    speed_limit = 130.0 / 3.6

    v_ego = state[0]
    v_front = state[1]
    distance_front = state[2]
    relative_velocity = state[3]
    cut_in_happened = state[4]
    remaining_distance = state[5]

    closing_speed = v_ego - v_front

    soft_distance = 5.0 + 0.6 * v_ego + 1.0 * max(0.0, closing_speed)
    hard_distance = 4.0 + 0.35 * v_ego + 0.80 * max(0.0, closing_speed)

    desired_distance = soft_distance + 3.0
    gap_error = distance_front - desired_distance

    after_cut_out = 1.0 if (
        env.cut_in_happened and not env.cut_in_active
    ) else 0.0

    cut_in_active = 1.0 if env.cut_in_active else 0.0

    return np.array(
        [
            v_ego / speed_limit,
            v_front / speed_limit,

            np.clip(distance_front / 300.0, 0.0, 5.0),
            np.clip(gap_error / 150.0, -5.0, 5.0),

            relative_velocity / speed_limit,
            closing_speed / speed_limit,

            soft_distance / 100.0,
            hard_distance / 100.0,

            cut_in_active,
            after_cut_out,

            remaining_distance / target_distance,

            env.prev_a_ego / 4.0,
        ],
        dtype=np.float32,
    )


def map_action_to_acceleration(action):
    """
    action = 0 means acceleration = 0.
    """

    action_value = float(np.clip(action[0], -1.0, 1.0))

    if action_value >= 0.0:
        acceleration = 2.0 * action_value
    else:
        acceleration = 4.0 * action_value

    return float(np.clip(acceleration, -4.0, 2.0))


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
        obs = normalize_state(state, env, target_distance=target_distance)

        action, _ = model.predict(obs, deterministic=True)

        # PPO output
        a_nom = map_action_to_acceleration(action)

        if use_cbf:
            a_cbf_cmd = cbf_three_vehicle_filter(a_nom, state, dt=env.dt)
        else:
            a_cbf_cmd = a_nom

        next_state, reward, done, info = env.step(a_cbf_cmd)

        info["a_nom"] = a_nom
        info["a_ppo_before_cbf"] = a_nom
        info["a_cbf_cmd"] = a_cbf_cmd
        info["cbf_correction"] = a_cbf_cmd - a_nom
        info["abs_cbf_correction"] = abs(a_cbf_cmd - a_nom)

        info["use_cbf"] = use_cbf
        info["method"] = "PPO + CBF" if use_cbf else "PPO without CBF"
        info["scenario"] = scenario

        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    if "a_cbf_cmd" in df.columns:
        correction = (df["a_cbf_cmd"] - df["a_nom"]).abs()
    else:
        correction = (df["a_ego"] - df["a_nom"]).abs()

    return {
        "reached_target": df["reached_target"].any(),
        "travel_time": df["time"].iloc[-1],
        "collision": df["collision"].any(),
        "safety_violations": df["safety_violation"].sum(),
        "cut_in_happened": df["cut_in_happened"].any(),
        "cut_out_happened": df["cut_in_left_this_step"].any(),
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

    # Keep these names so plot_three_vehicle.py can directly read them.
    ppo_df.to_csv("three_vehicle_baseline.csv", index=False)
    ppo_cbf_df.to_csv("three_vehicle_cbf.csv", index=False)

    print("PPO without CBF:")
    print(summarize_episode(ppo_df))

    print("\nPPO + CBF:")
    print(summarize_episode(ppo_cbf_df))

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

    print("\nSaved PPO and PPO+CBF evaluation results.")


if __name__ == "__main__":
    main()