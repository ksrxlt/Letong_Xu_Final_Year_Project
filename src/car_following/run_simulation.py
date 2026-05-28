import pandas as pd

from env import CarFollowingEnv
from controllers import rule_based_controller, cbf_safety_filter


def run_episode(use_cbf=False, seed=0):
    env = CarFollowingEnv()
    state = env.reset(seed=seed)

    logs = []
    done = False

    while not done:
        # Nominal action from controller
        a_nom = rule_based_controller(state)

        # Safety filter
        if use_cbf:
            a_ego = cbf_safety_filter(a_nom, state)
        else:
            a_ego = a_nom

        next_state, reward, done, info = env.step(a_ego)

        info["a_nom"] = a_nom
        info["use_cbf"] = use_cbf
        logs.append(info)

        state = next_state

    return pd.DataFrame(logs)


def summarize_episode(df):
    summary = {
        "collision": df["collision"].any(),
        "num_safety_violations": df["safety_violation"].sum(),
        "min_distance": df["distance"].min(),
        "mean_speed": df["v_ego"].mean(),
        "total_reward": df["reward"].sum(),
    }

    return summary


if __name__ == "__main__":
    baseline_df = run_episode(use_cbf=False, seed=1)
    cbf_df = run_episode(use_cbf=True, seed=1)

    baseline_summary = summarize_episode(baseline_df)
    cbf_summary = summarize_episode(cbf_df)

    print("Baseline result:")
    print(baseline_summary)

    print("\nCBF result:")
    print(cbf_summary)

    baseline_df.to_csv("baseline_result.csv", index=False)
    cbf_df.to_csv("cbf_result.csv", index=False)

    print("\nSaved results to baseline_result.csv and cbf_result.csv")