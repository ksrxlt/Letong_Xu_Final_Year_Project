import pandas as pd

from run_simulation import run_episode, summarize_episode


def evaluate_method(use_cbf=False, num_episodes=20):
    summaries = []

    for seed in range(num_episodes):
        df = run_episode(use_cbf=use_cbf, seed=seed)
        summary = summarize_episode(df)

        summary["seed"] = seed
        summary["method"] = "CBF" if use_cbf else "Baseline"

        summaries.append(summary)

    return pd.DataFrame(summaries)


def main():
    num_episodes = 20

    baseline_results = evaluate_method(use_cbf=False, num_episodes=num_episodes)
    cbf_results = evaluate_method(use_cbf=True, num_episodes=num_episodes)

    all_results = pd.concat([baseline_results, cbf_results], ignore_index=True)

    summary_table = all_results.groupby("method").agg(
        collision_rate=("collision", "mean"),
        avg_safety_violations=("num_safety_violations", "mean"),
        min_distance_mean=("min_distance", "mean"),
        min_distance_worst=("min_distance", "min"),
        mean_speed=("mean_speed", "mean"),
        avg_total_reward=("total_reward", "mean"),
    )

    print("\nRaw episode results:")
    print(all_results)

    print("\nSummary table:")
    print(summary_table)

    all_results.to_csv("all_episode_results.csv", index=False)
    summary_table.to_csv("summary_table.csv")

    print("\nSaved all_episode_results.csv and summary_table.csv")


if __name__ == "__main__":
    main()