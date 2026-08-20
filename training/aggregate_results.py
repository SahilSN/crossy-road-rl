import pandas as pd
from pathlib import Path

BENCHMARK_SEEDS = [0, 1, 2, 3, 4]

rows = []

# Only include official benchmark configurations.
official_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
]

for pattern in official_patterns:
    for path in Path("results/runs").glob(pattern):
        df = pd.read_csv(path)
        rows.append(df)

if not rows:
    raise SystemExit("No official evaluation.csv files found.")

all_results = pd.concat(rows, ignore_index=True)

# Keep only official benchmark training seeds.
all_results = all_results[
    all_results["training_seed"].isin(BENCHMARK_SEEDS)
].copy()

# Save only filtered benchmark runs.
all_results.to_csv(
    "results/benchmark_all_runs.csv",
    index=False,
)

summary = (
    all_results
    .groupby(["algorithm", "checkpoint_steps"])
    .agg(
        n_seeds=("training_seed", "nunique"),

        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),

        reward_mean=("avg_reward", "mean"),
        reward_std=("avg_reward", "std"),

        max_row_mean=("avg_max_row", "mean"),
        max_row_std=("avg_max_row", "std"),

        episode_length_mean=("avg_episode_length", "mean"),
        episode_length_std=("avg_episode_length", "std"),

        collision_rate_mean=("collision_rate", "mean"),
        collision_rate_std=("collision_rate", "std"),

        timeout_rate_mean=("timeout_rate", "mean"),
        timeout_rate_std=("timeout_rate", "std"),

        wait_pct_mean=("wait_pct", "mean"),
        forward_pct_mean=("forward_pct", "mean"),
        backward_pct_mean=("backward_pct", "mean"),
        left_pct_mean=("left_pct", "mean"),
        right_pct_mean=("right_pct", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    "results/benchmark_summary.csv",
    index=False,
)

print()
print("Saved:")
print("  results/benchmark_all_runs.csv")
print("  results/benchmark_summary.csv")

print()
print(summary.to_string(index=False))
