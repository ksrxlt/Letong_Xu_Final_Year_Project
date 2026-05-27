import pandas as pd

from multi_lane_env import MultiLaneCarFollowingEnv
from lane_change_controller import (
    rule_based_acceleration_controller,
    lane_change_decision,
    lane_change_safety_filter,
    longitudinal_safety_filter,
)


def run_episode(use_safety_filter=True, scenario="safe_overtake", seed=0):
    env = MultiLaneCarFollowingEnv(scenario=scenario)
    state = env.reset(seed=seed)

    logs = []
    done = False

    while not done:
        # Nominal longitudinal acceleration
        a_nom = rule_based_acceleration_controller(state)

        # Nominal lane-change command
        lane_cmd_nom = lane_change_decision(state)

        if use_safety_filter:
            a_ego = longitudinal_safety_filter(a_nom, state)
            lane_cmd = lane_change_safety_filter(lane_cmd_nom, state)
        else:
            a_ego = a_nom
            lane_cmd = lane_cmd_nom

        next_state, reward, done, info = env.step(
            a_ego=a_ego,
            lane_change_command=lane_cmd,
        )

        info["a_nom"] = a_nom
        info["lane_cmd_nom"] = lane_cmd_nom
        info["lane_cmd"] = lane_cmd
        info["use_safety_filter"] = use_safety_filter
        info["scenario"] = scenario

        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    return {
        "collision": df["collision"].any(),
        "num_safety_violations": df["safety_violation"].sum(),
        "unsafe_lane_change_attempts": df["unsafe_lane_change_attempt"].sum(),
        "lane_change_executed": df["lane_change_executed"].any(),
        "final_lane": df["lane_ego"].iloc[-1],
        "min_distance_front": df["distance_front"].min(),
        "mean_speed": df["v_ego"].mean(),
        "total_reward": df["reward"].sum(),
    }


def evaluate_scenario(scenario="safe_overtake", num_episodes=50):
    all_summaries = []

    for use_safety_filter in [False, True]:
        method = "Baseline" if not use_safety_filter else "Safety-filtered"

        for seed in range(num_episodes):
            df = run_episode(
                use_safety_filter=use_safety_filter,
                scenario=scenario,
                seed=seed,
            )

            summary = summarize_episode(df)
            summary["method"] = method
            summary["scenario"] = scenario
            summary["seed"] = seed
            all_summaries.append(summary)

    results = pd.DataFrame(all_summaries)

    summary_table = results.groupby("method").agg(
        collision_rate=("collision", "mean"),
        avg_safety_violations=("num_safety_violations", "mean"),
        avg_unsafe_lane_change_attempts=("unsafe_lane_change_attempts", "mean"),
        lane_change_rate=("lane_change_executed", "mean"),
        avg_min_distance_front=("min_distance_front", "mean"),
        mean_speed=("mean_speed", "mean"),
        avg_total_reward=("total_reward", "mean"),
    )

    return results, summary_table


def main():
    scenarios = [
        "safe_overtake",
        "blocked_by_front",
        "blocked_by_rear",
    ]

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")

        results, summary_table = evaluate_scenario(
            scenario=scenario,
            num_episodes=50,
        )

        print(summary_table)

        results.to_csv(f"lane_change_all_results_{scenario}.csv", index=False)
        summary_table.to_csv(f"lane_change_summary_{scenario}.csv")

        # Save one example trajectory for plotting
        baseline_df = run_episode(
            use_safety_filter=False,
            scenario=scenario,
            seed=1,
        )

        safe_df = run_episode(
            use_safety_filter=True,
            scenario=scenario,
            seed=1,
        )

        baseline_df.to_csv(f"lane_change_baseline_{scenario}.csv", index=False)
        safe_df.to_csv(f"lane_change_safe_{scenario}.csv", index=False)

    print("\nLane-change experiments completed.")


if __name__ == "__main__":
    main()
