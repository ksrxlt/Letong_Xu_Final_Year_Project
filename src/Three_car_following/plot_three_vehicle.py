import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASELINE_FILE = "three_vehicle_baseline.csv"   # PPO without CBF
CBF_FILE = "three_vehicle_cbf.csv"             # PPO + CBF


# ============================================================
# Basic utilities
# ============================================================

def load_df(filename):
    return pd.read_csv(filename)


def get_event_times(df):
    cut_in_time = None
    cut_out_time = None

    if "cut_in_this_step" in df.columns and df["cut_in_this_step"].astype(bool).any():
        cut_in_time = df.loc[df["cut_in_this_step"].astype(bool), "time"].iloc[0]

    if "cut_in_left_this_step" in df.columns and df["cut_in_left_this_step"].astype(bool).any():
        cut_out_time = df.loc[df["cut_in_left_this_step"].astype(bool), "time"].iloc[0]

    return cut_in_time, cut_out_time


def get_time_window(df, center_time, before=10.0, after=20.0):
    if center_time is None:
        return None

    start_time = max(df["time"].min(), center_time - before)
    end_time = min(df["time"].max(), center_time + after)

    return df[
        (df["time"] >= start_time)
        & (df["time"] <= end_time)
    ].copy()


def get_time_window_from_other_df(df, start_time, end_time):
    return df[
        (df["time"] >= start_time)
        & (df["time"] <= end_time)
    ].copy()


def add_single_event_line(event_time, label, linestyle):
    if event_time is not None:
        plt.axvline(
            event_time,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )


def add_event_lines(df):
    cut_in_time, cut_out_time = get_event_times(df)

    if cut_in_time is not None:
        plt.axvline(
            cut_in_time,
            linestyle=":",
            linewidth=2,
            label="Cut-in enters",
        )

    if cut_out_time is not None:
        plt.axvline(
            cut_out_time,
            linestyle="-.",
            linewidth=2,
            label="Cut-in leaves",
        )


# ============================================================
# Vehicle speed display
# ============================================================

def get_vehicle2_display_series(df):
    """
    Vehicle 2:
        hide before cut-in
        show during cut-in
        hide after cut-out
    """
    cut_in_time, cut_out_time = get_event_times(df)

    time = df["time"]
    v2 = df["v_cut_in"]

    v2_display = pd.Series(np.nan, index=df.index)

    if cut_in_time is None:
        return v2_display

    if cut_out_time is None:
        during_mask = time >= cut_in_time
    else:
        during_mask = (time >= cut_in_time) & (time < cut_out_time)

    v2_display[during_mask] = v2[during_mask]

    return v2_display


# ============================================================
# Common plotting helpers
# ============================================================

def plot_distance_lines(df):
    plt.plot(
        df["time"],
        df["distance_front"],
        linewidth=2,
        label="Current front distance",
    )

    if "soft_barrier_distance" in df.columns:
        plt.plot(
            df["time"],
            df["soft_barrier_distance"],
            "--",
            linewidth=2,
            label="Soft comfort barrier",
        )
    elif "safe_distance" in df.columns:
        plt.plot(
            df["time"],
            df["safe_distance"],
            "--",
            linewidth=2,
            label="Soft comfort barrier",
        )

    if "hard_barrier_distance" in df.columns:
        plt.plot(
            df["time"],
            df["hard_barrier_distance"],
            ":",
            linewidth=2,
            label="Hard safety barrier",
        )


def plot_speed_lines(baseline_df, cbf_df):
    speed_limit = 130.0 / 3.6

    # 车2只在 cut-in 存在期间显示
    v2_display = get_vehicle2_display_series(cbf_df)

    # baseline ego
    if len(baseline_df) > 0:
        plt.plot(
            baseline_df["time"],
            baseline_df["v_ego"],
            "--",
            linewidth=2,
            label="Ego speed PPO without CBF",
        )

    # cbf ego
    plt.plot(
        cbf_df["time"],
        cbf_df["v_ego"],
        linewidth=2,
        label="Ego speed PPO + CBF",
    )

    # 车1：全程显示
    plt.plot(
        cbf_df["time"],
        cbf_df["v_front"],
        linewidth=2,
        label="Vehicle 1 speed",
    )

    # 车2：仅 cut-in 存在期间显示
    plt.plot(
        cbf_df["time"],
        v2_display,
        linewidth=2,
        label="Vehicle 2 speed",
    )

    # 限速线
    plt.axhline(
        speed_limit,
        linestyle=":",
        linewidth=2,
        label="Speed limit 130 km/h",
    )


def plot_acceleration_lines(baseline_df, cbf_df):
    if len(baseline_df) > 0:
        plt.plot(
            baseline_df["time"],
            baseline_df["a_ego"],
            "--",
            linewidth=2,
            label="Actual acceleration PPO without CBF",
        )

    plt.plot(
        cbf_df["time"],
        cbf_df["a_ego"],
        linewidth=2,
        label="Actual acceleration PPO + CBF",
    )


def finish_plot(xlabel, ylabel, title, filename):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# ============================================================
# Full overview plots
# ============================================================

def plot_distance_overview():
    cbf_df = load_df(CBF_FILE)

    plt.figure(figsize=(11, 5))

    plot_distance_lines(cbf_df)
    add_event_lines(cbf_df)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title="Distance to Current Front Vehicle",
        filename="figure1_distance_to_current_front_vehicle.png",
    )


def plot_speed_overview():
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)

    plt.figure(figsize=(11, 5))

    plot_speed_lines(baseline_df, cbf_df)
    add_event_lines(cbf_df)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Speed (m/s)",
        title="Vehicle Speeds",
        filename="figure2_vehicle_speeds.png",
    )


# ============================================================
# Cut-in entry plots
# ============================================================

def plot_cut_in_entry_distance(before=12.0, after=25.0):
    cbf_df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(cbf_df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry distance plot.")
        return

    entry_df = get_time_window(cbf_df, cut_in_time, before=before, after=after)

    plt.figure(figsize=(11, 5))

    plot_distance_lines(entry_df)
    add_single_event_line(cut_in_time, "Cut-in enters", ":")

    finish_plot(
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title="Cut-in Entry Distance Response",
        filename="figure3_cut_in_entry_distance.png",
    )


def plot_cut_in_entry_speed(before=12.0, after=25.0):
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(cbf_df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry speed plot.")
        return

    entry_df = get_time_window(cbf_df, cut_in_time, before=before, after=after)

    start_time = entry_df["time"].min()
    end_time = entry_df["time"].max()
    baseline_entry_df = get_time_window_from_other_df(baseline_df, start_time, end_time)

    plt.figure(figsize=(11, 5))

    plot_speed_lines(baseline_entry_df, entry_df)
    add_single_event_line(cut_in_time, "Cut-in enters", ":")

    finish_plot(
        xlabel="Time (s)",
        ylabel="Speed (m/s)",
        title="Cut-in Entry Speed Response",
        filename="figure4_cut_in_entry_speed.png",
    )


def plot_cut_in_entry_acceleration(before=8.0, after=18.0):
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(cbf_df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry acceleration plot.")
        return

    entry_df = get_time_window(cbf_df, cut_in_time, before=before, after=after)

    start_time = entry_df["time"].min()
    end_time = entry_df["time"].max()
    baseline_entry_df = get_time_window_from_other_df(baseline_df, start_time, end_time)

    plt.figure(figsize=(11, 5))

    plot_acceleration_lines(baseline_entry_df, entry_df)
    add_single_event_line(cut_in_time, "Cut-in enters", ":")

    finish_plot(
        xlabel="Time (s)",
        ylabel="Acceleration (m/s²)",
        title="Cut-in Entry Acceleration Response",
        filename="figure5_cut_in_entry_acceleration.png",
    )


# ============================================================
# Cut-out leave plots
# ============================================================

def plot_cut_out_leave_distance(before=12.0, after=30.0, distance_ylim=None):
    cbf_df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(cbf_df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave distance plot.")
        return

    leave_df = get_time_window(cbf_df, cut_out_time, before=before, after=after)

    plt.figure(figsize=(11, 5))

    plot_distance_lines(leave_df)
    add_single_event_line(cut_out_time, "Cut-in leaves", "-.")

    if distance_ylim is not None:
        plt.ylim(distance_ylim)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title="Cut-out Recovery Distance Response",
        filename="figure6_cut_out_recovery_distance.png",
    )


def plot_cut_out_leave_speed(before=12.0, after=30.0):
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(cbf_df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave speed plot.")
        return

    leave_df = get_time_window(cbf_df, cut_out_time, before=before, after=after)

    start_time = leave_df["time"].min()
    end_time = leave_df["time"].max()
    baseline_leave_df = get_time_window_from_other_df(baseline_df, start_time, end_time)

    plt.figure(figsize=(11, 5))

    plot_speed_lines(baseline_leave_df, leave_df)
    add_single_event_line(cut_out_time, "Cut-in leaves", "-.")

    finish_plot(
        xlabel="Time (s)",
        ylabel="Speed (m/s)",
        title="Cut-out Recovery Speed Response",
        filename="figure7_cut_out_recovery_speed.png",
    )


def plot_cut_out_leave_acceleration(before=8.0, after=18.0):
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(cbf_df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave acceleration plot.")
        return

    leave_df = get_time_window(cbf_df, cut_out_time, before=before, after=after)

    start_time = leave_df["time"].min()
    end_time = leave_df["time"].max()
    baseline_leave_df = get_time_window_from_other_df(baseline_df, start_time, end_time)

    plt.figure(figsize=(11, 5))

    plot_acceleration_lines(baseline_leave_df, leave_df)
    add_single_event_line(cut_out_time, "Cut-in leaves", "-.")

    finish_plot(
        xlabel="Time (s)",
        ylabel="Acceleration (m/s²)",
        title="Cut-out Recovery Acceleration Response",
        filename="figure8_cut_out_recovery_acceleration.png",
    )


# ============================================================
# Optional: CBF intervention plot
# ============================================================

def plot_cbf_intervention():
    cbf_df = load_df(CBF_FILE)

    if "a_nom" not in cbf_df.columns or "a_ego" not in cbf_df.columns:
        print("No a_nom/a_ego columns found. Skip CBF intervention plot.")
        return

    correction = (cbf_df["a_ego"] - cbf_df["a_nom"]).abs()

    plt.figure(figsize=(11, 5))

    plt.plot(
        cbf_df["time"],
        cbf_df["a_nom"],
        "--",
        linewidth=2,
        label="PPO nominal acceleration a_nom",
    )

    plt.plot(
        cbf_df["time"],
        cbf_df["a_ego"],
        linewidth=2,
        label="CBF-filtered acceleration a_ego",
    )

    plt.plot(
        cbf_df["time"],
        correction,
        ":",
        linewidth=2,
        label="|CBF correction|",
    )

    add_event_lines(cbf_df)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Acceleration / correction (m/s²)",
        title="CBF Intervention: PPO Nominal vs Filtered Acceleration",
        filename="figure9_cbf_intervention.png",
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    plot_distance_overview()
    plot_speed_overview()

    plot_cut_in_entry_distance()
    plot_cut_in_entry_speed()
    plot_cut_in_entry_acceleration()

    plot_cut_out_leave_distance()
    plot_cut_out_leave_speed()
    plot_cut_out_leave_acceleration()

    plot_cbf_intervention()