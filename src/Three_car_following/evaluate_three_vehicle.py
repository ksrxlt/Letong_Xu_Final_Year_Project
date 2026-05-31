import pandas as pd

from run_three_vehicle import run_episode, summarize_episode


def evaluate_method(use_cbf=False, num_episodes=50, scenario="normal", target_distance=4000.0):
    summaries = []

    for seed in range(num_episodes):
        df = run_episode(
            use_cbf=use_cbf,
            seed=seed,
            scenario=scenario,
            target_distance=target_distance,
        )

        summary = summarize_episode(df)
        summary["seed"] = seed
        summary["method"] = "CBF" if use_cbf else "Baseline"
        summary["scenario"] = scenario

        summaries.append(summary)

    return pd.DataFrame(summaries)


def evaluate_scenario(scenario="normal", num_episodes=50, target_distance=4000.0):
    baseline_results = evaluate_method(
        use_cbf=False,
        num_episodes=num_episodes,
        scenario=scenario,
        target_distance=target_distance,
    )

    cbf_results = evaluate_method(
        use_cbf=True,
        num_episodes=num_episodes,
        scenario=scenario,
        target_distance=target_distance,
    )

    all_results = pd.concat([baseline_results, cbf_results], ignore_index=True)

    summary_table = all_results.groupby("method").agg(
        reached_target_rate=("reached_target", "mean"),
        mean_travel_time=("travel_time", "mean"),
        collision_rate=("collision", "mean"),
        avg_safety_violations=("safety_violations", "mean"),
        avg_too_far_steps=("too_far_steps", "mean"),
        cut_in_rate=("cut_in_happened", "mean"),
        mean_cut_in_time=("cut_in_time", "mean"),
        mean_min_front_distance=("min_front_distance", "mean"),
        worst_min_front_distance=("min_front_distance", "min"),
        mean_front_distance=("mean_front_distance", "mean"),
        mean_speed=("mean_speed", "mean"),
        avg_total_reward=("total_reward", "mean"),
        mean_abs_acceleration=("mean_abs_acceleration", "mean"),
    )

    return all_results, summary_table


def main():
    scenarios = ["normal", "hard"]

    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")

        all_results, summary_table = evaluate_scenario(
            scenario=scenario,
            num_episodes=50,
            target_distance=4000.0,
        )

        print(summary_table)

        all_results.to_csv(f"three_vehicle_all_results_{scenario}.csv", index=False)
        summary_table.to_csv(f"three_vehicle_summary_{scenario}.csv")

    print("\nCut-in evaluation completed.")


if __name__ == "__main__":
    main()