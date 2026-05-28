import pandas as pd
import matplotlib.pyplot as plt


def plot_distance():
    baseline_df = pd.read_csv("baseline_result.csv")
    cbf_df = pd.read_csv("cbf_result.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_df["time"],
        baseline_df["distance"],
        label="Baseline distance",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["distance"],
        label="CBF distance",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["safe_distance"],
        "--",
        label="Safe distance",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Inter-vehicle Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("distance_plot.png", dpi=300)
    plt.show()


def plot_velocity():
    baseline_df = pd.read_csv("baseline_result.csv")
    cbf_df = pd.read_csv("cbf_result.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_df["time"],
        baseline_df["v_ego"],
        label="Baseline ego velocity",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["v_ego"],
        label="CBF ego velocity",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["v_front"],
        "--",
        label="Front vehicle velocity",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Vehicle Velocities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("velocity_plot.png", dpi=300)
    plt.show()


def plot_acceleration():
    baseline_df = pd.read_csv("baseline_result.csv")
    cbf_df = pd.read_csv("cbf_result.csv")

    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_df["time"],
        baseline_df["a_ego"],
        label="Baseline acceleration",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["a_ego"],
        label="CBF acceleration",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Ego Vehicle Acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acceleration_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_distance()
    plot_velocity()
    plot_acceleration()