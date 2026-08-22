import pandas as pd
from pathlib import Path

BENCHMARK_SEEDS = [0, 1, 2, 3, 4]

rows = []

# ------------------------------------------------------------
# V3 official benchmark runs
# ------------------------------------------------------------

v3_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
    "a2c_seed*/evaluation.csv",
]

for pattern in v3_patterns:
    for path in Path("results/runs").glob(pattern):
        df = pd.read_csv(path)

        # Older v3 evaluation files may not yet contain this column.
        if "environment" not in df.columns:
            df["environment"] = "v3"

        rows.append(df)

# ------------------------------------------------------------
# V4 official benchmark runs
# ------------------------------------------------------------

v4_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
    "a2c_seed*/evaluation.csv",
]

for pattern in v4_patterns:
    for path in Path("results/runs/v4").glob(pattern):
        df = pd.read_csv(path)

        if "environment" not in df.columns:
            df["environment"] = "v4"

        rows.append(df)


# ------------------------------------------------------------
# V5 official benchmark runs
# ------------------------------------------------------------

v5_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
]

for pattern in v5_patterns:
    for path in Path("results/runs/v5").glob(pattern):
        df = pd.read_csv(path)

        if "environment" not in df.columns:
            df["environment"] = "v5"

        rows.append(df)


# ------------------------------------------------------------
# V6 observation-horizon benchmark runs
# ------------------------------------------------------------

v6_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
]

for environment in ["v6_local1", "v6_local3"]:
    for pattern in v6_patterns:
        for path in (Path("results/runs") / environment).glob(pattern):
            df = pd.read_csv(path)

            if "environment" not in df.columns:
                df["environment"] = environment

            rows.append(df)


# ------------------------------------------------------------
# V7 procedural-layout benchmark runs
# ------------------------------------------------------------

v7_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
]

for pattern in v7_patterns:
    for path in Path("results/runs/v7").glob(pattern):
        df = pd.read_csv(path)

        if "environment" not in df.columns:
            df["environment"] = "v7"

        rows.append(df)


# ------------------------------------------------------------
# V8 mixed road + river benchmark
# ------------------------------------------------------------

v8_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
]

for pattern in v8_patterns:
    for path in Path("results/runs/v8").glob(pattern):
        df = pd.read_csv(path)

        if "environment" not in df.columns:
            df["environment"] = "v8"

        rows.append(df)


# ------------------------------------------------------------
# V9 procedural mixed-mechanics benchmark
# ------------------------------------------------------------

v9_patterns = [
    "ppo_seed*/evaluation.csv",
    "dqn_seed*/evaluation.csv",
    "qrdqn_50q_seed*/evaluation.csv",
    "trpo_seed*/evaluation.csv",
]

for pattern in v9_patterns:
    for path in Path("results/runs/v9").glob(pattern):
        df = pd.read_csv(path)

        if "environment" not in df.columns:
            df["environment"] = "v9"

        rows.append(df)

if not rows:
    raise SystemExit("No official evaluation.csv files found.")

all_results = pd.concat(rows, ignore_index=True)

# Keep only official benchmark seeds.
all_results = all_results[
    all_results["training_seed"].isin(BENCHMARK_SEEDS)
].copy()

all_results = all_results.sort_values(
    [
        "environment",
        "algorithm",
        "checkpoint_steps",
        "training_seed",
    ]
)

all_results.to_csv(
    "results/benchmark_all_runs.csv",
    index=False,
)

summary = (
    all_results
    .groupby(
        [
            "environment",
            "algorithm",
            "checkpoint_steps",
        ]
    )
    .agg(
        n_seeds=("training_seed", "nunique"),

        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),

        reward_mean=("avg_reward", "mean"),
        reward_std=("avg_reward", "std"),

        max_row_mean=("avg_max_row", "mean"),
        max_row_std=("avg_max_row", "std"),

        episode_length_mean=(
            "avg_episode_length",
            "mean",
        ),
        episode_length_std=(
            "avg_episode_length",
            "std",
        ),

        collision_rate_mean=(
            "collision_rate",
            "mean",
        ),
        collision_rate_std=(
            "collision_rate",
            "std",
        ),

        timeout_rate_mean=(
            "timeout_rate",
            "mean",
        ),
        timeout_rate_std=(
            "timeout_rate",
            "std",
        ),

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

for environment in sorted(summary["environment"].unique()):
    print("=" * 110)
    print(f"ENVIRONMENT: {environment}")
    print("=" * 110)

    subset = summary[
        summary["environment"] == environment
    ]

    print(
        subset.to_string(
            index=False
        )
    )

    print()
