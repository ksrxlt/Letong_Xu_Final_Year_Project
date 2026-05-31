import pandas as pd

from env_three_vehicle import ThreeVehicleFollowingEnv
from controllers_three_vehicle import (
    rule_based_three_vehicle_controller,
    cbf_three_vehicle_filter,
)


def run_episode(use_cbf=False, seed=0, scenario="normal", target_distance=10000.0):
    env = ThreeVehicleFollowingEnv(
        scenario=scenario,
        target_distance=target_distance,
    )

    state = env.reset(seed=seed)

    logs = []
    done = False

    while not done:
        a_nom = rule_based_three_vehicle_controller(state)

        if use_cbf:
            a_ego = cbf_three_vehicle_filter(a_nom, state)
        else:
            a_ego = a_nom

        next_state, reward, done, info = env.step(a_ego)

        info["a_nom"] = a_nom
        info["use_cbf"] = use_cbf
        info["scenario"] = scenario

        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    return {
        "reached_target": df["reached_target"].any(),
        "travel_time": df["time"].iloc[-1],
        "collision": df["collision"].any(),
        "safety_violations": df["safety_violation"].sum(),
        "too_far_steps": df["too_far"].sum(),
        "cut_in_happened": df["cut_in_happened"].any(),
        "cut_in_time": df["cut_in_time"].dropna().iloc[0]
        if df["cut_in_time"].notna().any()
        else None,
        "min_front_distance": df["distance_front"].min(),
        "mean_front_distance": df["distance_front"].mean(),
        "mean_speed": df["v_ego"].mean(),
        "total_reward": df["reward"].sum(),
        "mean_abs_acceleration": df["a_ego"].abs().mean(),
    }


if __name__ == "__main__":
    baseline_df = run_episode(
        use_cbf=False,
        seed=1,
        scenario="normal",
        target_distance=4000.0,
    )

    cbf_df = run_episode(
        use_cbf=True,
        seed=1,
        scenario="normal",
        target_distance=4000.0,
    )

    baseline_df.to_csv("three_vehicle_baseline.csv", index=False)
    cbf_df.to_csv("three_vehicle_cbf.csv", index=False)

    print("Baseline:")
    print(summarize_episode(baseline_df))

    print("\nCBF:")
    print(summarize_episode(cbf_df))

    print("\nSaved three_vehicle_baseline.csv and three_vehicle_cbf.csv")