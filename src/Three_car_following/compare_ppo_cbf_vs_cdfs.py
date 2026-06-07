"""
Replay comparison: PPO + CBF vs CDFS
=====================================

Purpose
-------
Use an already successful PPO + CBF rollout stored in `three_vehicle_cbf.csv`
as the fixed external traffic scenario, then simulate a traditional CDFS
(Constant Distance Following Strategy) controller on the same Vehicle 1 / Vehicle 2
trajectory.

This avoids the unfair comparison problem where PPO + CBF and CDFS generate
different cut-in / cut-out timings or different front-vehicle trajectories.

Outputs
-------
comparison_results_replay/
    comparison_raw_data.csv
    comparison_metrics.csv
    comparison_phase_metrics.csv
    comparison_winner_summary.csv
    comparison_score_summary.csv
    comparison_distance.png
    comparison_gap_error_soft.png
    comparison_speed.png
    comparison_acceleration.png
    comparison_metric_bars.png
    ppo_cbf_intervention.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

REFERENCE_CSV = "three_vehicle_cbf.csv"
RESULT_DIR = "comparison_results_replay"
os.makedirs(RESULT_DIR, exist_ok=True)

# The same target distance used in your PPO evaluation.
TARGET_DISTANCE = 4000.0

# Speed limit: 130 km/h.
SPEED_LIMIT = 130.0 / 3.6

# Ego acceleration limits.
A_MIN = -4.0
A_MAX = 2.0

# CDFS controller parameters.
# A reasonable fixed-distance baseline. You can tune these, but keep them fixed
# once you decide the final baseline.
CDFS_DESIRED_DISTANCE = 22.0
CDFS_K_GAP = 0.12
CDFS_K_REL_SPEED = 0.75

# Actuator smoothing for CDFS. This mirrors the idea that the baseline should
# also be physically reasonable and not instantly jump acceleration.
CDFS_ACCEL_SMOOTHING_ALPHA = 0.25
CDFS_MAX_DELTA_A_NORMAL = 0.15
CDFS_MAX_DELTA_A_EMERGENCY = 0.60

# Plot settings.
FIGSIZE = (14, 6)
DPI = 300


# ============================================================
# Utility helpers
# ============================================================

def first_existing_column(df, names, required=True, default=None):
    """Return the first column name found in df from a candidate list."""
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"None of these columns exist in CSV: {names}")
    return default


def as_bool_series(s):
    """Robust conversion of CSV boolean/string/0-1 columns to bool."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) > 0.5
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def compute_dt(ref):
    t = ref["time"].to_numpy(dtype=float)
    if len(t) < 2:
        return 0.1
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return 0.1
    return float(dt)


def soft_barrier_distance(v_ego, v_front_eff):
    """Same soft comfort barrier structure as env_three_vehicle.py."""
    min_gap = 5.0
    closing_speed = max(0.0, v_ego - v_front_eff)
    return min_gap + 0.6 * v_ego + 1.0 * closing_speed


def hard_barrier_distance(v_ego, v_front_eff):
    """Same hard safety barrier structure as env_three_vehicle.py."""
    min_gap = 4.0
    closing_speed = max(0.0, v_ego - v_front_eff)
    return min_gap + 0.35 * v_ego + 0.80 * closing_speed


def get_cut_times(df):
    cut_in_time = np.nan
    cut_out_time = np.nan

    if "cut_in_this_step" in df.columns:
        rows = df[as_bool_series(df["cut_in_this_step"])]
        if len(rows) > 0:
            cut_in_time = float(rows["time"].iloc[0])

    if not np.isfinite(cut_in_time) and "cut_in_active" in df.columns:
        active = as_bool_series(df["cut_in_active"])
        idx = np.flatnonzero(active.to_numpy())
        if len(idx) > 0:
            cut_in_time = float(df["time"].iloc[idx[0]])

    if "cut_in_left_this_step" in df.columns:
        rows = df[as_bool_series(df["cut_in_left_this_step"])]
        if len(rows) > 0:
            cut_out_time = float(rows["time"].iloc[0])

    if not np.isfinite(cut_out_time) and "cut_in_active" in df.columns:
        active = as_bool_series(df["cut_in_active"]).to_numpy()
        changes = np.flatnonzero((active[:-1] == True) & (active[1:] == False))
        if len(changes) > 0:
            cut_out_time = float(df["time"].iloc[changes[0] + 1])

    return cut_in_time, cut_out_time


def savefig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=DPI)
    plt.show()


# ============================================================
# Load and standardise reference PPO + CBF rollout
# ============================================================

def load_reference_rollout(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Put this script in the same folder as three_vehicle_cbf.csv, "
            "or change REFERENCE_CSV at the top of this file."
        )

    ref = pd.read_csv(path)

    # Normalise important column names.
    if "soft_distance" not in ref.columns:
        soft_col = first_existing_column(
            ref,
            ["soft_barrier_distance", "safe_distance", "soft_distance"],
            required=True,
        )
        ref["soft_distance"] = ref[soft_col]

    if "hard_distance" not in ref.columns:
        hard_col = first_existing_column(
            ref,
            ["hard_barrier_distance", "hard_distance"],
            required=True,
        )
        ref["hard_distance"] = ref[hard_col]

    if "time" not in ref.columns:
        raise KeyError("Reference CSV must contain a 'time' column.")

    # Required external trajectory columns.
    required_cols = ["x_front", "v_front", "x_cut_in", "v_cut_in", "x_ego", "v_ego", "distance_front"]
    missing = [c for c in required_cols if c not in ref.columns]
    if missing:
        raise KeyError(
            "The replay comparison requires these columns in three_vehicle_cbf.csv: "
            f"{missing}. Re-run evaluate_rl_three_vehicle.py after ensuring env info records x_front, "
            "v_front, x_cut_in, v_cut_in, x_ego, v_ego, and distance_front."
        )

    if "cut_in_active" not in ref.columns:
        raise KeyError("Reference CSV must contain 'cut_in_active' for replay.")

    if "a_ego" not in ref.columns:
        raise KeyError("Reference CSV must contain 'a_ego'.")

    # Fill optional columns.
    if "a_nom" not in ref.columns:
        ref["a_nom"] = ref["a_ego"]
    if "a_cbf_cmd" not in ref.columns:
        ref["a_cbf_cmd"] = ref["a_ego"]
    if "abs_cbf_correction" not in ref.columns:
        if "cbf_correction" in ref.columns:
            ref["abs_cbf_correction"] = ref["cbf_correction"].abs()
        else:
            ref["abs_cbf_correction"] = (ref["a_cbf_cmd"] - ref["a_nom"]).abs()

    if "collision" not in ref.columns:
        ref["collision"] = False
    if "reached_target" not in ref.columns:
        ref["reached_target"] = ref["x_ego"] >= TARGET_DISTANCE
    if "safety_violation" not in ref.columns:
        ref["safety_violation"] = ref["distance_front"] < ref["hard_distance"]

    return ref.copy()


def build_ppo_cbf_from_reference(ref):
    """Use the existing three_vehicle_cbf.csv as the PPO + CBF result."""
    df = ref.copy()
    df["method"] = "PPO + CBF"
    df["cbf_correction"] = df["abs_cbf_correction"]
    df["energy_accel_sq_step"] = df["a_ego"] ** 2
    return df


# ============================================================
# CDFS replay on the same external traffic trajectory
# ============================================================

def cdfs_controller(distance_front, v_ego, v_front_eff):
    """
    Constant Distance Following Strategy.

    Target:
        distance_front -> CDFS_DESIRED_DISTANCE

    Control law:
        a = K_gap * (gap - desired_gap) + K_rel * (v_front - v_ego)
    """
    gap_error = distance_front - CDFS_DESIRED_DISTANCE
    relative_speed = v_front_eff - v_ego
    a_cmd = CDFS_K_GAP * gap_error + CDFS_K_REL_SPEED * relative_speed
    return float(np.clip(a_cmd, A_MIN, A_MAX))


def apply_actuator_model(a_cmd, prev_a, distance_front, hard_distance, soft_distance, closing_speed):
    """Apply the same style of acceleration smoothing used in the environment."""
    a_cmd = float(np.clip(a_cmd, A_MIN, A_MAX))

    a_target = prev_a + CDFS_ACCEL_SMOOTHING_ALPHA * (a_cmd - prev_a)

    emergency_brake_needed = (
        distance_front < hard_distance + 3.0
        or (distance_front < soft_distance + 5.0 and closing_speed > 3.0)
    )

    if emergency_brake_needed and a_target < prev_a:
        max_delta = CDFS_MAX_DELTA_A_EMERGENCY
    else:
        max_delta = CDFS_MAX_DELTA_A_NORMAL

    delta = np.clip(a_target - prev_a, -max_delta, max_delta)
    a_exec = prev_a + delta
    return float(np.clip(a_exec, A_MIN, A_MAX))


def simulate_cdfs_on_reference(ref):
    """Simulate CDFS ego vehicle using the same external vehicle trajectories."""
    dt = compute_dt(ref)

    x_ego = float(ref["x_ego"].iloc[0])
    v_ego = float(ref["v_ego"].iloc[0])
    prev_a = 0.0

    logs = []

    cut_active_series = as_bool_series(ref["cut_in_active"]).to_numpy()

    for i, row in ref.iterrows():
        t = float(row["time"])

        x_front = float(row["x_front"])
        v_front = float(row["v_front"])
        x_cut_in = float(row["x_cut_in"])
        v_cut_in = float(row["v_cut_in"])
        cut_in_active = bool(cut_active_series[i])

        # Same external-front selection rule as the original environment:
        # during cut-in, use vehicle 2 if it is in front; otherwise use vehicle 1.
        if cut_in_active and x_cut_in > x_ego:
            front_type = "cut_in"
            x_front_eff = x_cut_in
            v_front_eff = v_cut_in
        elif x_front > x_ego:
            front_type = "front"
            x_front_eff = x_front
            v_front_eff = v_front
        else:
            front_type = "none"
            x_front_eff = x_ego + 999.0
            v_front_eff = v_ego

        distance_front = float(x_front_eff - x_ego)
        closing_speed = float(v_ego - v_front_eff)

        soft_distance = soft_barrier_distance(v_ego, v_front_eff)
        hard_distance = hard_barrier_distance(v_ego, v_front_eff)

        a_cmd = cdfs_controller(distance_front, v_ego, v_front_eff)
        a_ego = apply_actuator_model(
            a_cmd=a_cmd,
            prev_a=prev_a,
            distance_front=distance_front,
            hard_distance=hard_distance,
            soft_distance=soft_distance,
            closing_speed=closing_speed,
        )

        # Log current step before updating to next position, matching usual env logging style.
        collision = distance_front <= 0.0
        safety_violation = distance_front < hard_distance
        reached_target = x_ego >= TARGET_DISTANCE

        logs.append({
            "method": "CDFS",
            "time": t,
            "x_ego": x_ego,
            "v_ego": v_ego,
            "a_ego": a_ego,
            "a_nom": a_cmd,
            "a_cbf_cmd": a_cmd,
            "cbf_correction": 0.0,
            "abs_cbf_correction": 0.0,
            "x_front": x_front,
            "v_front": v_front,
            "x_cut_in": x_cut_in,
            "v_cut_in": v_cut_in,
            "front_type": front_type,
            "distance_front": distance_front,
            "v_front_eff": v_front_eff,
            "soft_distance": soft_distance,
            "hard_distance": hard_distance,
            "soft_barrier_distance": soft_distance,
            "hard_barrier_distance": hard_distance,
            "gap_error_to_soft": distance_front - soft_distance,
            "safety_violation": safety_violation,
            "collision": collision,
            "reached_target": reached_target,
            "cut_in_active": cut_in_active,
            "cut_in_this_step": bool(row["cut_in_this_step"]) if "cut_in_this_step" in ref.columns else False,
            "cut_in_left_this_step": bool(row["cut_in_left_this_step"]) if "cut_in_left_this_step" in ref.columns else False,
            "reward": np.nan,
        })

        if reached_target or collision:
            # Continue logging could make arrival-time comparison less clean.
            # Break once the CDFS ego reaches the same target or crashes.
            break

        # Update ego dynamics for next step.
        v_ego = max(0.0, min(SPEED_LIMIT, v_ego + a_ego * dt))
        x_ego = x_ego + v_ego * dt
        prev_a = a_ego

    return pd.DataFrame(logs)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(df):
    rows = []

    for method, g in df.groupby("method"):
        g = g.copy().sort_values("time")
        if len(g) == 0:
            continue

        dt = compute_dt(g)

        collision = bool(as_bool_series(g["collision"]).any()) if "collision" in g.columns else False
        reached_target = bool(as_bool_series(g["reached_target"]).any()) if "reached_target" in g.columns else False

        if reached_target:
            arrival_time = float(g.loc[as_bool_series(g["reached_target"]), "time"].iloc[0])
        else:
            arrival_time = float(g["time"].iloc[-1])

        gap_error_soft = g["distance_front"].to_numpy(float) - g["soft_distance"].to_numpy(float)
        hard_margin = g["distance_front"].to_numpy(float) - g["hard_distance"].to_numpy(float)
        soft_violation = np.maximum(0.0, -gap_error_soft)
        safety_violation = hard_margin < 0.0

        a = g["a_ego"].to_numpy(float)
        if len(a) > 1:
            jerk = np.diff(a) / dt
        else:
            jerk = np.array([0.0])

        if "cbf_correction" in g.columns:
            correction = np.abs(g["cbf_correction"].to_numpy(float))
        elif "abs_cbf_correction" in g.columns:
            correction = np.abs(g["abs_cbf_correction"].to_numpy(float))
        else:
            correction = np.zeros(len(g))

        rows.append({
            "method": method,
            "collision": collision,
            "reached_target": reached_target,
            "arrival_time_s": arrival_time,
            "num_steps": len(g),
            "mean_speed": np.nanmean(g["v_ego"]),
            "max_speed": np.nanmax(g["v_ego"]),
            "mean_distance_front_m": np.nanmean(g["distance_front"]),
            "min_distance_front_m": np.nanmin(g["distance_front"]),
            "mean_abs_gap_error_to_soft_m": np.nanmean(np.abs(gap_error_soft)),
            "rmse_gap_error_to_soft_m": np.sqrt(np.nanmean(gap_error_soft ** 2)),
            "mean_gap_error_to_soft_m": np.nanmean(gap_error_soft),
            "min_hard_margin_m": np.nanmin(hard_margin),
            "safety_violation_rate": np.nanmean(safety_violation),
            "mean_soft_violation_m": np.nanmean(soft_violation),
            "max_soft_violation_m": np.nanmax(soft_violation),
            "energy_accel_sq": np.nansum((a ** 2) * dt),
            "energy_positive_accel_sq": np.nansum((np.maximum(a, 0.0) ** 2) * dt),
            "mean_abs_accel": np.nanmean(np.abs(a)),
            "accel_rms": np.sqrt(np.nanmean(a ** 2)),
            "jerk_abs_mean": np.nanmean(np.abs(jerk)),
            "jerk_rms": np.sqrt(np.nanmean(jerk ** 2)),
            "cbf_mean_correction": np.nanmean(correction),
            "cbf_max_correction": np.nanmax(correction),
            "cbf_intervention_rate": np.nanmean(correction > 0.01),
        })

    return pd.DataFrame(rows)


def compute_phase_metrics(df):
    rows = []

    for method, g in df.groupby("method"):
        g = g.copy().sort_values("time")
        if len(g) < 2:
            continue

        cut_in_time, cut_out_time = get_cut_times(g)

        phases = []
        if np.isfinite(cut_in_time):
            phases.append(("before_cut_in", g[g["time"] < cut_in_time]))
        if np.isfinite(cut_in_time) and np.isfinite(cut_out_time):
            phases.append(("during_cut_in", g[(g["time"] >= cut_in_time) & (g["time"] < cut_out_time)]))
        if np.isfinite(cut_out_time):
            phases.append(("after_cut_out", g[g["time"] >= cut_out_time]))

        for phase_name, p in phases:
            if len(p) < 2:
                continue

            dt = compute_dt(p)
            gap_error_soft = p["distance_front"].to_numpy(float) - p["soft_distance"].to_numpy(float)
            hard_margin = p["distance_front"].to_numpy(float) - p["hard_distance"].to_numpy(float)
            a = p["a_ego"].to_numpy(float)
            jerk = np.diff(a) / dt if len(a) > 1 else np.array([0.0])

            rows.append({
                "method": method,
                "phase": phase_name,
                "num_steps": len(p),
                "mean_abs_gap_error_to_soft_m": np.nanmean(np.abs(gap_error_soft)),
                "rmse_gap_error_to_soft_m": np.sqrt(np.nanmean(gap_error_soft ** 2)),
                "mean_gap_error_to_soft_m": np.nanmean(gap_error_soft),
                "min_hard_margin_m": np.nanmin(hard_margin),
                "safety_violation_rate": np.nanmean(hard_margin < 0.0),
                "mean_speed": np.nanmean(p["v_ego"]),
                "max_speed": np.nanmax(p["v_ego"]),
                "mean_abs_accel": np.nanmean(np.abs(a)),
                "accel_rms": np.sqrt(np.nanmean(a ** 2)),
                "jerk_abs_mean": np.nanmean(np.abs(jerk)),
                "jerk_rms": np.sqrt(np.nanmean(jerk ** 2)),
                "energy_accel_sq": np.nansum((a ** 2) * dt),
            })

    return pd.DataFrame(rows)


# ============================================================
# Winner / score summary
# ============================================================

def build_winner_summary(metrics_df):
    if len(metrics_df) < 2:
        return pd.DataFrame()

    m = metrics_df.set_index("method")
    if "PPO + CBF" not in m.index or "CDFS" not in m.index:
        return pd.DataFrame()

    ppo = m.loc["PPO + CBF"]
    cdfs = m.loc["CDFS"]

    rows = []

    def add_metric(category, metric, lower_is_better=True):
        ppo_value = ppo[metric]
        cdfs_value = cdfs[metric]

        if isinstance(ppo_value, (bool, np.bool_)) or isinstance(cdfs_value, (bool, np.bool_)):
            ppo_score_value = int(bool(ppo_value))
            cdfs_score_value = int(bool(cdfs_value))
        else:
            ppo_score_value = float(ppo_value)
            cdfs_score_value = float(cdfs_value)

        if lower_is_better:
            if np.isclose(ppo_score_value, cdfs_score_value, equal_nan=True):
                winner = "Tie"
            elif ppo_score_value < cdfs_score_value:
                winner = "PPO + CBF"
            else:
                winner = "CDFS"
            diff = cdfs_score_value - ppo_score_value
            baseline = abs(cdfs_score_value) if abs(cdfs_score_value) > 1e-9 else np.nan
        else:
            if np.isclose(ppo_score_value, cdfs_score_value, equal_nan=True):
                winner = "Tie"
            elif ppo_score_value > cdfs_score_value:
                winner = "PPO + CBF"
            else:
                winner = "CDFS"
            diff = ppo_score_value - cdfs_score_value
            baseline = abs(cdfs_score_value) if abs(cdfs_score_value) > 1e-9 else np.nan

        improvement_pct = 100.0 * diff / baseline if np.isfinite(baseline) else np.nan

        rows.append({
            "category": category,
            "metric": metric,
            "ppo_cbf": ppo_value,
            "cdfs": cdfs_value,
            "winner": winner,
            "difference_positive_means_ppo_better": diff,
            "ppo_improvement_over_cdfs_percent": improvement_pct,
            "better_when": "lower" if lower_is_better else "higher",
        })

    # Safety.
    add_metric("Safety", "collision", lower_is_better=True)
    add_metric("Safety", "safety_violation_rate", lower_is_better=True)
    add_metric("Safety", "min_hard_margin_m", lower_is_better=False)

    # Efficiency.
    add_metric("Efficiency", "arrival_time_s", lower_is_better=True)
    add_metric("Efficiency", "mean_speed", lower_is_better=False)
    add_metric("Efficiency", "energy_accel_sq", lower_is_better=True)

    # Distance keeping.
    add_metric("Distance keeping", "mean_abs_gap_error_to_soft_m", lower_is_better=True)
    add_metric("Distance keeping", "rmse_gap_error_to_soft_m", lower_is_better=True)
    add_metric("Distance keeping", "mean_soft_violation_m", lower_is_better=True)

    # Comfort.
    add_metric("Comfort", "mean_abs_accel", lower_is_better=True)
    add_metric("Comfort", "accel_rms", lower_is_better=True)
    add_metric("Comfort", "jerk_abs_mean", lower_is_better=True)
    add_metric("Comfort", "jerk_rms", lower_is_better=True)

    return pd.DataFrame(rows)


def build_score_summary(winner_summary_df):
    if len(winner_summary_df) == 0:
        return pd.DataFrame()

    rows = []
    for category, g in winner_summary_df.groupby("category"):
        ppo_score = 0.0
        cdfs_score = 0.0

        for _, row in g.iterrows():
            if row["winner"] == "PPO + CBF":
                ppo_score += 1.0
            elif row["winner"] == "CDFS":
                cdfs_score += 1.0
            else:
                ppo_score += 0.5
                cdfs_score += 0.5

        if np.isclose(ppo_score, cdfs_score):
            winner = "Tie"
        elif ppo_score > cdfs_score:
            winner = "PPO + CBF"
        else:
            winner = "CDFS"

        rows.append({
            "category": category,
            "ppo_cbf_score": ppo_score,
            "cdfs_score": cdfs_score,
            "winner": winner,
        })

    total_ppo = sum(r["ppo_cbf_score"] for r in rows)
    total_cdfs = sum(r["cdfs_score"] for r in rows)

    overall_winner = "Tie" if np.isclose(total_ppo, total_cdfs) else ("PPO + CBF" if total_ppo > total_cdfs else "CDFS")
    rows.append({
        "category": "Overall",
        "ppo_cbf_score": total_ppo,
        "cdfs_score": total_cdfs,
        "winner": overall_winner,
    })

    return pd.DataFrame(rows)


def print_human_readable_summary(winner_summary_df, score_summary_df):
    print("\n================ Winner Summary ================")
    if len(winner_summary_df) == 0:
        print("Winner summary is empty. Check method names in metrics_df.")
        return

    for _, row in winner_summary_df.iterrows():
        print(f"[{row['category']}] {row['metric']}")
        print(f"  PPO + CBF: {row['ppo_cbf']}")
        print(f"  CDFS:      {row['cdfs']}")
        print(f"  Winner:    {row['winner']}")

        improve = row["ppo_improvement_over_cdfs_percent"]
        if isinstance(improve, (float, int, np.floating)) and np.isfinite(improve):
            if improve >= 0:
                print(f"  PPO improvement over CDFS: {improve:.2f}%")
            else:
                print(f"  PPO worse than CDFS by: {abs(improve):.2f}%")
        print("")

    print("================ Score Summary ================")
    print(score_summary_df.to_string(index=False))
    print("================================================\n")


# ============================================================
# Plotting
# ============================================================

def plot_results(all_df, ref):
    cut_in_time, cut_out_time = get_cut_times(ref)

    # Reference barriers from PPO + CBF reference rollout.
    ref_time = ref["time"]
    ref_soft = ref["soft_distance"]
    ref_hard = ref["hard_distance"]

    methods = list(all_df["method"].unique())

    # --------------------------------------------------------
    # Distance comparison
    # --------------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for method in methods:
        g = all_df[all_df["method"] == method]
        plt.plot(g["time"], g["distance_front"], label=f"{method}: current gap")

    plt.plot(ref_time, ref_soft, "--", label="Soft comfort barrier")
    plt.plot(ref_time, ref_hard, ":", label="Hard safety barrier")
    if np.isfinite(cut_in_time):
        plt.axvline(cut_in_time, linestyle=":", label="Cut-in enters")
    if np.isfinite(cut_out_time):
        plt.axvline(cut_out_time, linestyle="-.", label="Cut-in leaves")

    plt.title("Distance Keeping Comparison: PPO + CBF vs CDFS")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend()
    savefig("comparison_distance.png")

    # --------------------------------------------------------
    # Gap error to soft barrier
    # --------------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for method in methods:
        g = all_df[all_df["method"] == method]
        gap_error = g["distance_front"] - g["soft_distance"]
        plt.plot(g["time"], gap_error, label=f"{method}: gap - soft")

    plt.axhline(0.0, linestyle="--", label="Soft barrier reference")
    if np.isfinite(cut_in_time):
        plt.axvline(cut_in_time, linestyle=":", label="Cut-in enters")
    if np.isfinite(cut_out_time):
        plt.axvline(cut_out_time, linestyle="-.", label="Cut-in leaves")

    plt.title("Gap Error to Soft Barrier")
    plt.xlabel("Time (s)")
    plt.ylabel("Gap - Soft Barrier (m)")
    plt.grid(True)
    plt.legend()
    savefig("comparison_gap_error_soft.png")

    # --------------------------------------------------------
    # Ego speed comparison
    # --------------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for method in methods:
        g = all_df[all_df["method"] == method]
        plt.plot(g["time"], g["v_ego"], label=f"{method}: ego speed")

    plt.axhline(SPEED_LIMIT, linestyle=":", label="Speed limit 130 km/h")
    if np.isfinite(cut_in_time):
        plt.axvline(cut_in_time, linestyle=":", label="Cut-in enters")
    if np.isfinite(cut_out_time):
        plt.axvline(cut_out_time, linestyle="-.", label="Cut-in leaves")

    plt.title("Ego Speed Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    savefig("comparison_speed.png")

    # --------------------------------------------------------
    # Acceleration comparison
    # --------------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    for method in methods:
        g = all_df[all_df["method"] == method]
        plt.plot(g["time"], g["a_ego"], label=f"{method}: executed acceleration")

    if np.isfinite(cut_in_time):
        plt.axvline(cut_in_time, linestyle=":", label="Cut-in enters")
    if np.isfinite(cut_out_time):
        plt.axvline(cut_out_time, linestyle="-.", label="Cut-in leaves")

    plt.title("Executed Acceleration Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True)
    plt.legend()
    savefig("comparison_acceleration.png")

    # --------------------------------------------------------
    # PPO CBF intervention only
    # --------------------------------------------------------
    plt.figure(figsize=FIGSIZE)
    g = all_df[all_df["method"] == "PPO + CBF"]
    if len(g) > 0:
        plt.plot(g["time"], g["cbf_correction"], label="PPO + CBF: |CBF command - PPO output|")
    if np.isfinite(cut_in_time):
        plt.axvline(cut_in_time, linestyle=":", label="Cut-in enters")
    if np.isfinite(cut_out_time):
        plt.axvline(cut_out_time, linestyle="-.", label="Cut-in leaves")

    plt.title("CBF Intervention for PPO Policy")
    plt.xlabel("Time (s)")
    plt.ylabel("CBF correction (m/s²)")
    plt.grid(True)
    plt.legend()
    savefig("ppo_cbf_intervention.png")


def plot_metric_bar_summary(winner_summary_df):
    selected_metrics = [
        "arrival_time_s",
        "mean_abs_gap_error_to_soft_m",
        "safety_violation_rate",
        "energy_accel_sq",
        "mean_abs_accel",
        "jerk_rms",
    ]

    plot_df = winner_summary_df[winner_summary_df["metric"].isin(selected_metrics)].copy()
    if len(plot_df) == 0:
        return

    labels = plot_df["metric"].tolist()
    ppo_values = plot_df["ppo_cbf"].astype(float).to_numpy()
    cdfs_values = plot_df["cdfs"].astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=FIGSIZE)
    plt.bar(x - width / 2, ppo_values, width, label="PPO + CBF")
    plt.bar(x + width / 2, cdfs_values, width, label="CDFS")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Metric value")
    plt.title("Key Metric Comparison: PPO + CBF vs CDFS")
    plt.grid(True, axis="y")
    plt.legend()
    savefig("comparison_metric_bars.png")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading reference PPO + CBF rollout:", REFERENCE_CSV)
    ref = load_reference_rollout(REFERENCE_CSV)

    cut_in_time, cut_out_time = get_cut_times(ref)
    print(f"Reference cut-in time:  {cut_in_time}")
    print(f"Reference cut-out time: {cut_out_time}")
    print(f"Reference max gap:      {ref['distance_front'].max():.3f} m")
    print(f"Reference end time:     {ref['time'].iloc[-1]:.3f} s")

    ppo_df = build_ppo_cbf_from_reference(ref)
    cdfs_df = simulate_cdfs_on_reference(ref)

    all_df = pd.concat([ppo_df, cdfs_df], ignore_index=True)

    raw_path = os.path.join(RESULT_DIR, "comparison_raw_data.csv")
    metrics_path = os.path.join(RESULT_DIR, "comparison_metrics.csv")
    phase_metrics_path = os.path.join(RESULT_DIR, "comparison_phase_metrics.csv")
    winner_summary_path = os.path.join(RESULT_DIR, "comparison_winner_summary.csv")
    score_summary_path = os.path.join(RESULT_DIR, "comparison_score_summary.csv")

    all_df.to_csv(raw_path, index=False)

    metrics_df = compute_metrics(all_df)
    metrics_df.to_csv(metrics_path, index=False)

    phase_metrics_df = compute_phase_metrics(all_df)
    phase_metrics_df.to_csv(phase_metrics_path, index=False)

    winner_summary_df = build_winner_summary(metrics_df)
    winner_summary_df.to_csv(winner_summary_path, index=False)

    score_summary_df = build_score_summary(winner_summary_df)
    score_summary_df.to_csv(score_summary_path, index=False)

    print("\n================ Overall Metrics ================")
    print(metrics_df.to_string(index=False))
    print("=================================================\n")

    print("\n================ Phase Metrics ================")
    print(phase_metrics_df.to_string(index=False))
    print("===============================================\n")

    print_human_readable_summary(winner_summary_df, score_summary_df)

    plot_results(all_df, ref)
    plot_metric_bar_summary(winner_summary_df)

    print("Saved raw data to:          ", raw_path)
    print("Saved metrics to:           ", metrics_path)
    print("Saved phase metrics to:     ", phase_metrics_path)
    print("Saved winner summary to:    ", winner_summary_path)
    print("Saved score summary to:     ", score_summary_path)
    print("Saved figures to folder:    ", RESULT_DIR)


if __name__ == "__main__":
    main()
