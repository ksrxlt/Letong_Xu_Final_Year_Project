import pandas as pd
import matplotlib.pyplot as plt


def plot_lane_trajectory(scenario):
    baseline_df = pd.read_csv(f"lane_change_baseline_{scenario}.csv")
    safe_df = pd.read_csv(f"lane_change_safe_{scenario}.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_df["time"],
        baseline_df["lane_ego"],
        label="Baseline ego lane",
    )

    plt.plot(
        safe_df["time"],
        safe_df["lane_ego"],
        label="Safety-filtered ego lane",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Lane")
    plt.title(f"Ego Lane over Time - {scenario}")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lane_trajectory_{scenario}.png", dpi=300)
    plt.show()


def plot_positions(scenario):
    safe_df = pd.read_csv(f"lane_change_safe_{scenario}.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        safe_df["time"],
        safe_df["x_ego"],
        label="Ego vehicle",
    )

    plt.plot(
        safe_df["time"],
        safe_df["x_slow"],
        label="Slow vehicle",
    )

    plt.plot(
        safe_df["time"],
        safe_df["x_side"],
        label="Side vehicle",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Longitudinal position x (m)")
    plt.title(f"Vehicle Positions - {scenario}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"positions_{scenario}.png", dpi=300)
    plt.show()


def plot_distance_front(scenario):
    baseline_df = pd.read_csv(f"lane_change_baseline_{scenario}.csv")
    safe_df = pd.read_csv(f"lane_change_safe_{scenario}.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_df["time"],
        baseline_df["distance_front"],
        label="Baseline distance to front vehicle",
    )

    plt.plot(
        safe_df["time"],
        safe_df["distance_front"],
        label="Safety-filtered distance to front vehicle",
    )

    plt.plot(
        safe_df["time"],
        safe_df["safe_distance"],
        "--",
        label="Safe distance",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title(f"Distance to Front Vehicle - {scenario}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"distance_front_{scenario}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    scenarios = [
        "safe_overtake",
        "blocked_by_front",
        "blocked_by_rear",
    ]

    for scenario in scenarios:
        plot_lane_trajectory(scenario)
        plot_positions(scenario)
        plot_distance_front(scenario)