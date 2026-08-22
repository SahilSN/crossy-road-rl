from pathlib import Path

import pandas as pd


INPUT_DIR = Path("results/cross_env_transfer")
OUTPUT_DIR = Path("results/cross_env_transfer")

ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]
ENVIRONMENTS = ["v8", "v9"]


# ------------------------------------------------------------
# Load transfer CSVs
# ------------------------------------------------------------

rows = []

for train_env in ENVIRONMENTS:
    for eval_env in ENVIRONMENTS:
        if train_env == eval_env:
            continue

        for algorithm in ALGORITHMS:
            for seed in range(5):
                path = (
                    INPUT_DIR
                    / (
                        f"train_{train_env}"
                        f"_eval_{eval_env}"
                        f"_{algorithm}"
                        f"_seed{seed}.csv"
                    )
                )

                if not path.exists():
                    print(f"Missing: {path}")
                    continue

                df = pd.read_csv(path)

                if len(df) != 1:
                    raise RuntimeError(
                        f"Expected 1 row in {path}, "
                        f"found {len(df)}"
                    )

                rows.append(df.iloc[0].to_dict())


raw = pd.DataFrame(rows)


# ------------------------------------------------------------
# Validate completeness
# ------------------------------------------------------------

print()
print("CROSS-ENVIRONMENT TRANSFER SEED COUNT")
print("=" * 80)

counts = (
    raw.groupby(
        ["train_env", "eval_env", "algorithm"]
    )["training_seed"]
    .nunique()
)

print(counts.to_string())

if (counts != 5).any():
    print()
    print(
        "WARNING: At least one transfer condition "
        "does not contain all five seeds."
    )


# ------------------------------------------------------------
# Aggregate transfer performance
# ------------------------------------------------------------

summary = (
    raw.groupby(
        ["train_env", "eval_env", "algorithm"]
    )
    .agg(
        n_seeds=("training_seed", "nunique"),

        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),

        reward_mean=("avg_reward", "mean"),
        reward_std=("avg_reward", "std"),

        max_row_mean=("avg_max_row", "mean"),
        max_row_std=("avg_max_row", "std"),

        road_collision_mean=(
            "road_collision_rate",
            "mean",
        ),

        drowning_mean=(
            "drowning_rate",
            "mean",
        ),

        timeout_mean=(
            "timeout_rate",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("CROSS-ENVIRONMENT TRANSFER SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        formatters={
            "success_mean": lambda x: f"{x:.1%}",
            "success_std": lambda x: f"{x:.1%}",
            "reward_mean": lambda x: f"{x:.3f}",
            "reward_std": lambda x: f"{x:.3f}",
            "max_row_mean": lambda x: f"{x:.2f}",
            "max_row_std": lambda x: f"{x:.2f}",
            "road_collision_mean": lambda x: f"{x:.1%}",
            "drowning_mean": lambda x: f"{x:.1%}",
            "timeout_mean": lambda x: f"{x:.1%}",
        },
    )
)


# ------------------------------------------------------------
# Load native v8/v9 1M baselines
# ------------------------------------------------------------

benchmark = pd.read_csv(
    "results/benchmark_all_runs.csv"
)

native = benchmark[
    (benchmark["checkpoint_steps"] == 1_000_000)
    & (benchmark["environment"].isin(["v8", "v9"]))
    & (benchmark["algorithm"].isin(ALGORITHMS))
].copy()

native_summary = (
    native.groupby(
        ["environment", "algorithm"]
    )
    .agg(
        native_success_mean=("success_rate", "mean"),
        native_success_std=("success_rate", "std"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Compare transfer against native performance of training env
# ------------------------------------------------------------

comparison = summary.merge(
    native_summary,
    left_on=["train_env", "algorithm"],
    right_on=["environment", "algorithm"],
    how="left",
)

comparison["retained_fraction"] = (
    comparison["success_mean"]
    / comparison["native_success_mean"]
)

comparison["absolute_drop"] = (
    comparison["success_mean"]
    - comparison["native_success_mean"]
)


print()
print("TRANSFER RETENTION RELATIVE TO TRAINING-ENV BASELINE")
print("=" * 110)

print(
    comparison[
        [
            "train_env",
            "eval_env",
            "algorithm",
            "native_success_mean",
            "success_mean",
            "absolute_drop",
            "retained_fraction",
        ]
    ].to_string(
        index=False,
        formatters={
            "native_success_mean":
                lambda x: f"{x:.1%}",

            "success_mean":
                lambda x: f"{x:.1%}",

            "absolute_drop":
                lambda x: f"{x:+.1%}",

            "retained_fraction":
                lambda x: f"{x:.2f}x",
        },
    )
)


# ------------------------------------------------------------
# Compact transfer matrix
# ------------------------------------------------------------

matrix = summary.copy()

matrix["display"] = matrix.apply(
    lambda row: (
        f"{row['success_mean'] * 100:.1f}% "
        f"± {row['success_std'] * 100:.1f}%"
    ),
    axis=1,
)

matrix["direction"] = (
    matrix["train_env"].str.upper()
    + " → "
    + matrix["eval_env"].str.upper()
)

pivot = matrix.pivot(
    index="direction",
    columns="algorithm",
    values="display",
).reindex(
    columns=ALGORITHMS,
)

pivot.columns = [
    "PPO",
    "TRPO",
    "DQN",
    "QR-DQN",
]

print()
print("TRANSFER MATRIX")
print("=" * 90)

print(pivot.to_string())


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

raw_path = (
    OUTPUT_DIR
    / "cross_env_transfer_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "cross_env_transfer_summary.csv"
)

comparison_path = (
    OUTPUT_DIR
    / "cross_env_transfer_retention.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

comparison.to_csv(
    comparison_path,
    index=False,
)

print()
print("Saved:")
print(f"  {raw_path}")
print(f"  {summary_path}")
print(f"  {comparison_path}")
