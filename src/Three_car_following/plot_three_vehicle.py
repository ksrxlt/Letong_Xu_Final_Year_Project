import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CBF_FILE = "three_vehicle_cbf.csv"


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
# Vehicle display utilities
# ============================================================

def get_vehicle2_display_series(df):
    """
    Vehicle 2:
        hide before cut-in
        show during cut-in
        hide after cut-out
    """

    v2_display = pd.Series(np.nan, index=df.index)

    if "v_cut_in" not in df.columns:
        return v2_display

    if "cut_in_active" in df.columns:
        mask = df["cut_in_active"].astype(bool)
        v2_display.loc[mask] = df.loc[mask, "v_cut_in"]
        return v2_display

    # Fallback if cut_in_active is not available
    cut_in_time, cut_out_time = get_event_times(df)
    time = df["time"]

    if cut_in_time is None:
        return v2_display

    if cut_out_time is None:
        mask = time >= cut_in_time
    else:
        mask = (time >= cut_in_time) & (time < cut_out_time)

    v2_display.loc[mask] = df.loc[mask, "v_cut_in"]

    return v2_display


# ============================================================
# Common plot helpers
# ============================================================

def plot_distance_lines(df, x=None):
    if x is None:
        x = df["time"]

    if "distance_front" in df.columns:
        plt.plot(
            x,
            df["distance_front"],
            linewidth=2,
            label="Current front distance",
        )

    if "soft_barrier_distance" in df.columns:
        plt.plot(
            x,
            df["soft_barrier_distance"],
            "--",
            linewidth=2,
            label="Soft comfort barrier",
        )
    elif "safe_distance" in df.columns:
        plt.plot(
            x,
            df["safe_distance"],
            "--",
            linewidth=2,
            label="Soft comfort barrier",
        )

    if "hard_barrier_distance" in df.columns:
        plt.plot(
            x,
            df["hard_barrier_distance"],
            ":",
            linewidth=2,
            label="Hard safety barrier",
        )


def plot_speed_lines(df, x=None):
    """
    Plot speed lines for one PPO+CBF rollout.

    Vehicle 1 is always plotted using v_front.
    Vehicle 2 is only plotted during cut-in active period using v_cut_in.
    """

    speed_limit = 130.0 / 3.6

    if x is None:
        x = df["time"]

    # Ego speed
    if "v_ego" in df.columns:
        plt.plot(
            x,
            df["v_ego"],
            linewidth=2,
            label="Ego speed",
        )

    # Vehicle 1 speed: always show
    if "v_front" in df.columns:
        plt.plot(
            x,
            df["v_front"],
            linewidth=2,
            label="Vehicle 1 speed",
        )

    # Vehicle 2 speed: only show during cut-in active period
    if "v_cut_in" in df.columns:
        v2_display = get_vehicle2_display_series(df)

        plt.plot(
            x,
            v2_display,
            linewidth=2,
            label="Vehicle 2 speed",
        )

    # Speed limit
    plt.axhline(
        speed_limit,
        linestyle=":",
        linewidth=2,
        label="Speed limit 130 km/h",
    )


def plot_control_lines(df, x=None):
    """
    Plot control signals from one PPO+CBF rollout.

    a_nom:
        PPO output before CBF.

    a_cbf_cmd:
        CBF output before env.step.

    a_ego:
        Final executed acceleration after env.step / actuator smoothing.
    """

    if x is None:
        x = df["time"]

    if "a_nom" in df.columns:
        plt.plot(
            x,
            df["a_nom"],
            "--",
            linewidth=2,
            label="PPO output before CBF",
        )

    if "a_cbf_cmd" in df.columns:
        plt.plot(
            x,
            df["a_cbf_cmd"],
            linewidth=2,
            label="CBF output command",
        )

    if "a_ego" in df.columns:
        plt.plot(
            x,
            df["a_ego"],
            ":",
            linewidth=2,
            label="Final executed acceleration",
        )


# ============================================================
# Full overview plots
# ============================================================

def plot_distance_overview():
    df = load_df(CBF_FILE)

    plt.figure(figsize=(11, 5))

    plot_distance_lines(df)
    add_event_lines(df)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title="Distance to Current Front Vehicle",
        filename="figure1_distance_to_current_front_vehicle.png",
    )


def plot_speed_overview():
    df = load_df(CBF_FILE)

    plt.figure(figsize=(11, 5))

    plot_speed_lines(df)
    add_event_lines(df)

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
    df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry distance plot.")
        return

    entry_df = get_time_window(
        df,
        cut_in_time,
        before=before,
        after=after,
    )

    if entry_df is None or len(entry_df) == 0:
        print("Empty cut-in entry distance window.")
        return

    tau = entry_df["time"] - cut_in_time

    plt.figure(figsize=(11, 5))

    plot_distance_lines(entry_df, x=tau)

    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=2,
        label="Cut-in enters",
    )

    finish_plot(
        xlabel="Time relative to cut-in entry (s)",
        ylabel="Distance (m)",
        title="Cut-in Entry Distance Response",
        filename="figure3_cut_in_entry_distance.png",
    )


def plot_cut_in_entry_speed(before=12.0, after=25.0):
    df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry speed plot.")
        return

    entry_df = get_time_window(
        df,
        cut_in_time,
        before=before,
        after=after,
    )

    if entry_df is None or len(entry_df) == 0:
        print("Empty cut-in entry speed window.")
        return

    tau = entry_df["time"] - cut_in_time

    plt.figure(figsize=(11, 5))

    plot_speed_lines(entry_df, x=tau)

    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=2,
        label="Cut-in enters",
    )

    finish_plot(
        xlabel="Time relative to cut-in entry (s)",
        ylabel="Speed (m/s)",
        title="Cut-in Entry Speed Response",
        filename="figure4_cut_in_entry_speed.png",
    )


def plot_cut_in_entry_acceleration(before=8.0, after=18.0):
    df = load_df(CBF_FILE)
    cut_in_time, _ = get_event_times(df)

    if cut_in_time is None:
        print("No cut-in event found. Skip cut-in entry acceleration plot.")
        return

    entry_df = get_time_window(
        df,
        cut_in_time,
        before=before,
        after=after,
    )

    if entry_df is None or len(entry_df) == 0:
        print("Empty cut-in entry acceleration window.")
        return

    tau = entry_df["time"] - cut_in_time

    plt.figure(figsize=(11, 5))

    plot_control_lines(entry_df, x=tau)

    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=2,
        label="Cut-in enters",
    )

    finish_plot(
        xlabel="Time relative to cut-in entry (s)",
        ylabel="Acceleration (m/s²)",
        title="Cut-in Entry: PPO Output vs CBF Output",
        filename="figure5_cut_in_entry_acceleration.png",
    )


# ============================================================
# Cut-out recovery plots
# ============================================================

def plot_cut_out_leave_distance(before=12.0, after=30.0, distance_ylim=None):
    df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave distance plot.")
        return

    leave_df = get_time_window(
        df,
        cut_out_time,
        before=before,
        after=after,
    )

    if leave_df is None or len(leave_df) == 0:
        print("Empty cut-out leave distance window.")
        return

    tau = leave_df["time"] - cut_out_time

    plt.figure(figsize=(11, 5))

    plot_distance_lines(leave_df, x=tau)

    plt.axvline(
        0.0,
        linestyle="-.",
        linewidth=2,
        label="Cut-in leaves",
    )

    if distance_ylim is not None:
        plt.ylim(distance_ylim)

    finish_plot(
        xlabel="Time relative to cut-out leave (s)",
        ylabel="Distance (m)",
        title="Cut-out Recovery Distance Response",
        filename="figure6_cut_out_recovery_distance.png",
    )


def plot_cut_out_leave_speed(before=12.0, after=30.0):
    df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave speed plot.")
        return

    leave_df = get_time_window(
        df,
        cut_out_time,
        before=before,
        after=after,
    )

    if leave_df is None or len(leave_df) == 0:
        print("Empty cut-out leave speed window.")
        return

    tau = leave_df["time"] - cut_out_time

    plt.figure(figsize=(11, 5))

    plot_speed_lines(leave_df, x=tau)

    plt.axvline(
        0.0,
        linestyle="-.",
        linewidth=2,
        label="Cut-in leaves",
    )

    finish_plot(
        xlabel="Time relative to cut-out leave (s)",
        ylabel="Speed (m/s)",
        title="Cut-out Recovery Speed Response",
        filename="figure7_cut_out_recovery_speed.png",
    )


def plot_cut_out_leave_acceleration(before=8.0, after=18.0):
    df = load_df(CBF_FILE)
    _, cut_out_time = get_event_times(df)

    if cut_out_time is None:
        print("No cut-out event found. Skip cut-out leave acceleration plot.")
        return

    leave_df = get_time_window(
        df,
        cut_out_time,
        before=before,
        after=after,
    )

    if leave_df is None or len(leave_df) == 0:
        print("Empty cut-out leave acceleration window.")
        return

    tau = leave_df["time"] - cut_out_time

    plt.figure(figsize=(11, 5))

    plot_control_lines(leave_df, x=tau)

    plt.axvline(
        0.0,
        linestyle="-.",
        linewidth=2,
        label="Cut-in leaves",
    )

    finish_plot(
        xlabel="Time relative to cut-out leave (s)",
        ylabel="Acceleration (m/s²)",
        title="Cut-out Recovery: PPO Output vs CBF Output",
        filename="figure8_cut_out_recovery_acceleration.png",
    )


# ============================================================
# Full CBF intervention plot
# ============================================================

def plot_cbf_intervention():
    df = load_df(CBF_FILE)

    if "a_nom" not in df.columns:
        print("No a_nom column found. Skip CBF intervention plot.")
        return

    if "a_cbf_cmd" in df.columns:
        correction = (df["a_cbf_cmd"] - df["a_nom"]).abs()
        correction_label = "|CBF command - PPO output|"
    else:
        correction = (df["a_ego"] - df["a_nom"]).abs()
        correction_label = "|Final acceleration - PPO output|"

    plt.figure(figsize=(11, 5))

    if "a_nom" in df.columns:
        plt.plot(
            df["time"],
            df["a_nom"],
            "--",
            linewidth=2,
            label="PPO output before CBF",
        )

    if "a_cbf_cmd" in df.columns:
        plt.plot(
            df["time"],
            df["a_cbf_cmd"],
            linewidth=2,
            label="CBF output command",
        )

    if "a_ego" in df.columns:
        plt.plot(
            df["time"],
            df["a_ego"],
            ":",
            linewidth=2,
            label="Final executed acceleration",
        )

    plt.plot(
        df["time"],
        correction,
        "-.",
        linewidth=2,
        label=correction_label,
    )

    add_event_lines(df)

    finish_plot(
        xlabel="Time (s)",
        ylabel="Acceleration / correction (m/s²)",
        title="CBF Intervention: PPO Output vs CBF Output",
        filename="figure9_cbf_intervention.png",
    )


# ============================================================
# Diagnostics
# ============================================================

def print_cbf_diagnostics():
    df = load_df(CBF_FILE)

    print("\n========== CBF Diagnostics ==========")

    if "a_nom" in df.columns and "a_cbf_cmd" in df.columns:
        increases = (df["a_cbf_cmd"] > df["a_nom"] + 1e-6).sum()
        correction = (df["a_cbf_cmd"] - df["a_nom"]).abs()

        print("CBF increases acceleration count:", increases)
        print("Mean |CBF correction|:", correction.mean())
        print("Max |CBF correction|:", correction.max())
        print("CBF intervention rate:", (correction > 0.01).mean())

    if "a_ego" in df.columns:
        delta_a = df["a_ego"].diff().abs()

        print("Mean |delta final acceleration|:", delta_a.mean())
        print("Max |delta final acceleration|:", delta_a.max())
        print("Large final acceleration jumps > 0.3:", (delta_a > 0.3).sum())

    if "distance_front" in df.columns and "soft_barrier_distance" in df.columns:
        e = df["distance_front"] - df["soft_barrier_distance"]

        print("Mean distance minus soft barrier:", e.mean())
        print("Mean absolute distance-soft error:", e.abs().mean())
        print("Within 3m of soft barrier rate:", (e.abs() < 3.0).mean())
        print("Within 5m of soft barrier rate:", (e.abs() < 5.0).mean())

    if "distance_front" in df.columns and "hard_barrier_distance" in df.columns:
        h = df["distance_front"] - df["hard_barrier_distance"]

        print("Minimum hard barrier margin:", h.min())
        print("Hard barrier violation count:", (h < 0.0).sum())

    print("=====================================\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print_cbf_diagnostics()

    plot_distance_overview()
    plot_speed_overview()

    plot_cut_in_entry_distance()
    plot_cut_in_entry_speed()
    plot_cut_in_entry_acceleration()

    plot_cut_out_leave_distance()
    plot_cut_out_leave_speed()
    plot_cut_out_leave_acceleration()

    plot_cbf_intervention()