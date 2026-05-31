import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASELINE_FILE = "three_vehicle_baseline.csv"
CBF_FILE = "three_vehicle_cbf.csv"


# =========================
# Basic helpers
# =========================
def load_df(filename):
    return pd.read_csv(filename)


def first_existing_column(df, candidates):
    """
    Return the first column name that exists in df from candidates.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns were found: {candidates}")


def get_front_distance_col(df):
    return first_existing_column(df, ["distance_front", "effective_front_distance"])


def get_safe_distance_col(df):
    return first_existing_column(df, ["safe_distance", "safe_front_distance"])


def get_nominal_acc_col(df):
    return first_existing_column(df, ["a_nom", "nominal_acceleration"])


def get_actual_acc_col(df):
    return first_existing_column(df, ["a_ego", "filtered_acceleration", "actual_acceleration"])


def get_front_speed_col(df):
    return first_existing_column(df, ["v_front", "v_vehicle1", "vehicle1_speed"])


def get_cut_in_speed_col(df):
    return first_existing_column(df, ["v_cut_in", "v_vehicle2", "vehicle2_speed"])


def get_time_col(df):
    return first_existing_column(df, ["time"])


def get_cut_in_enter_col(df):
    return first_existing_column(df, ["cut_in_this_step", "cut_in_enter_this_step", "cut_in_entered"])


def get_cut_in_leave_col(df):
    return first_existing_column(df, ["cut_in_left_this_step", "cut_in_leave_this_step", "cut_in_left"])


# =========================
# Event helpers
# =========================
def get_event_times(df):
    """
    Return cut-in enter time and leave time if they exist.
    """
    time_col = get_time_col(df)

    cut_in_time = None
    cut_out_time = None

    try:
        enter_col = get_cut_in_enter_col(df)
        if df[enter_col].astype(bool).any():
            cut_in_time = df.loc[df[enter_col].astype(bool), time_col].iloc[0]
    except KeyError:
        pass

    try:
        leave_col = get_cut_in_leave_col(df)
        if df[leave_col].astype(bool).any():
            cut_out_time = df.loc[df[leave_col].astype(bool), time_col].iloc[0]
    except KeyError:
        pass

    return cut_in_time, cut_out_time


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


# =========================
# Speed display logic
# =========================
def get_vehicle_speed_display_series(df):
    """
    Vehicle 1:
        show before cut-in
        hide during cut-in
        show again after cut-out

    Vehicle 2:
        hide before cut-in
        show during cut-in
        hide after cut-out
    """
    time_col = get_time_col(df)
    v1_col = get_front_speed_col(df)
    v2_col = get_cut_in_speed_col(df)

    t = df[time_col]
    v1 = df[v1_col]
    v2 = df[v2_col]

    cut_in_time, cut_out_time = get_event_times(df)

    v1_display = pd.Series(np.nan, index=df.index)
    v2_display = pd.Series(np.nan, index=df.index)

    if cut_in_time is None:
        # If no cut-in event recorded, only show vehicle 1
        v1_display[:] = v1
        return v1_display, v2_display

    # Before cut-in -> show vehicle 1
    before_mask = t < cut_in_time
    v1_display[before_mask] = v1[before_mask]

    # During cut-in -> show vehicle 2
    if cut_out_time is None:
        during_mask = t >= cut_in_time
    else:
        during_mask = (t >= cut_in_time) & (t < cut_out_time)

    v2_display[during_mask] = v2[during_mask]

    # After cut-out -> show vehicle 1 again
    if cut_out_time is not None:
        after_mask = t >= cut_out_time
        v1_display[after_mask] = v1[after_mask]

    return v1_display, v2_display


# =========================
# Plot 1: full distance
# =========================
def plot_distance():
    cbf_df = load_df(CBF_FILE)

    time_col = get_time_col(cbf_df)
    dist_col = get_front_distance_col(cbf_df)
    safe_col = get_safe_distance_col(cbf_df)

    plt.figure(figsize=(11, 5))

    plt.plot(
        cbf_df[time_col],
        cbf_df[dist_col],
        linewidth=2,
        label="Current front distance",
    )

    plt.plot(
        cbf_df[time_col],
        cbf_df[safe_col],
        "--",
        linewidth=2,
        label="Safe distance",
    )

    add_event_lines(cbf_df)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance to Current Front Vehicle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure1_distance_to_current_front_vehicle.png", dpi=300)
    plt.show()


# =========================
# Plot 2: speed
# =========================
def plot_speed():
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)

    time_col_base = get_time_col(baseline_df)
    time_col_cbf = get_time_col(cbf_df)

    v1_display, v2_display = get_vehicle_speed_display_series(cbf_df)

    speed_limit = 130.0 / 3.6  # 36.11 m/s

    plt.figure(figsize=(11, 5))

    plt.plot(
        baseline_df[time_col_base],
        baseline_df["v_ego"],
        "--",
        linewidth=2,
        label="Ego speed (Baseline)",
    )

    plt.plot(
        cbf_df[time_col_cbf],
        cbf_df["v_ego"],
        linewidth=2,
        label="Ego speed (CBF)",
    )

    plt.plot(
        cbf_df[time_col_cbf],
        v1_display,
        linewidth=2,
        label="Vehicle 1 speed",
    )

    plt.plot(
        cbf_df[time_col_cbf],
        v2_display,
        linewidth=2,
        label="Vehicle 2 speed",
    )

    plt.axhline(
        speed_limit,
        linestyle=":",
        linewidth=2,
        label="Speed limit 130 km/h",
    )

    add_event_lines(cbf_df)

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Vehicle Speeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure2_vehicle_speeds.png", dpi=300)
    plt.show()


# =========================
# Plot 3: zoomed distance
# =========================
def plot_distance_zoom():
    cbf_df = load_df(CBF_FILE)

    time_col = get_time_col(cbf_df)
    dist_col = get_front_distance_col(cbf_df)
    safe_col = get_safe_distance_col(cbf_df)

    cut_in_time, cut_out_time = get_event_times(cbf_df)

    if cut_in_time is None:
        print("No cut-in event found. Skip zoomed distance plot.")
        return

    start_time = max(0.0, cut_in_time - 20.0)

    if cut_out_time is None:
        end_time = cut_in_time + 80.0
    else:
        end_time = cut_out_time + 40.0

    zoom_df = cbf_df[
        (cbf_df[time_col] >= start_time)
        & (cbf_df[time_col] <= end_time)
    ].copy()

    plt.figure(figsize=(11, 5))

    plt.plot(
        zoom_df[time_col],
        zoom_df[dist_col],
        linewidth=2,
        label="Current front distance",
    )

    plt.plot(
        zoom_df[time_col],
        zoom_df[safe_col],
        "--",
        linewidth=2,
        label="Safe distance",
    )

    add_event_lines(cbf_df)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Zoomed Distance Response around Cut-in and Cut-out")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure3_zoomed_distance_response.png", dpi=300)
    plt.show()


# =========================
# Optional Plot 4: CBF correction
# =========================
def plot_cbf_correction():
    cbf_df = load_df(CBF_FILE)

    time_col = get_time_col(cbf_df)
    a_nom_col = get_nominal_acc_col(cbf_df)
    a_ego_col = get_actual_acc_col(cbf_df)

    correction = cbf_df[a_ego_col] - cbf_df[a_nom_col]

    plt.figure(figsize=(11, 5))

    plt.plot(
        cbf_df[time_col],
        correction,
        linewidth=2,
        label="CBF correction: a_ego - a_nom",
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="No correction",
    )

    add_event_lines(cbf_df)

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration correction (m/s²)")
    plt.title("CBF Intervention on Nominal Acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure4_cbf_acceleration_correction.png", dpi=300)
    plt.show()

def plot_acceleration_zoom():
    baseline_df = load_df(BASELINE_FILE)
    cbf_df = load_df(CBF_FILE)

    time_col_base = get_time_col(baseline_df)
    time_col_cbf = get_time_col(cbf_df)

    cut_in_time, cut_out_time = get_event_times(cbf_df)

    if cut_in_time is None:
        print("No cut-in event found. Skip zoomed acceleration plot.")
        return

    # Show from shortly before cut-in to shortly after cut-out
    start_time = max(0.0, cut_in_time - 8.0)

    if cut_out_time is None:
        end_time = cut_in_time + 20.0
        title = "Acceleration around Cut-in Entry"
    else:
        end_time = cut_out_time + 8.0
        title = "Acceleration around Cut-in Entry and Exit"

    baseline_zoom = baseline_df[
        (baseline_df[time_col_base] >= start_time)
        & (baseline_df[time_col_base] <= end_time)
    ].copy()

    cbf_zoom = cbf_df[
        (cbf_df[time_col_cbf] >= start_time)
        & (cbf_df[time_col_cbf] <= end_time)
    ].copy()

    plt.figure(figsize=(11, 5))

    plt.plot(
        baseline_zoom[time_col_base],
        baseline_zoom["a_ego"],
        "--",
        linewidth=2,
        label="Actual acceleration without CBF",
    )

    plt.plot(
        cbf_zoom[time_col_cbf],
        cbf_zoom["a_ego"],
        linewidth=2,
        label="Actual acceleration with CBF",
    )

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

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure4_zoomed_actual_acceleration_baseline_vs_cbf.png", dpi=300)
    plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    plot_distance()
    plot_speed()
    plot_distance_zoom()
    plot_acceleration_zoom()

    # If you still want to check how much CBF really modifies action,
    # uncomment the next line:
    # plot_cbf_correction()