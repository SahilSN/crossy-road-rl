from pathlib import Path

import pandas as pd


RUN_ROOT = Path("results/runs")
OUTPUT_DIR = Path("results/figures/domain_randomization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]
ENVIRONMENTS = ["v9", "v10"]
SPEEDS = [0.8, 1.0, 1.2, 1.4]


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"
    return f"{algorithm}_seed{seed}"


def speed_tag(speed):
    return str(speed).replace(".", "p")


def eval_filename(speed):
    if speed == 1.0:
        return "evaluation.csv"
    return f"evaluation_speed_{speed_tag(speed)}.csv"


rows = []

for env_name in ENVIRONMENTS:
    for algorithm in ALGORITHMS:
        for seed in range(5):
            run_dir = (
                RUN_ROOT
                / env_name
                / run_name(algorithm, seed)
            )

            for speed in SPEEDS:
                path = (
                    run_dir
                    / eval_filename(speed)
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing evaluation file: {path}"
                    )

                df = pd.read_csv(path)

                final = df[
                    df["checkpoint_steps"]
                    == 1_000_000
                ]

                if len(final) != 1:
                    raise RuntimeError(
                        f"Expected exactly one 1M row in {path}, "
                        f"found {len(final)}"
                    )

                row = final.iloc[0]

                rows.append({
                    "environment": env_name,
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "speed_scale": speed,
                    "success_rate": row["success_rate"],
                    "avg_reward": row["avg_reward"],
                    "avg_max_row": row["avg_max_row"],
                    "road_collision_rate": row["road_collision_rate"],
                    "drowning_rate": row["drowning_rate"],
                    "timeout_rate": row["timeout_rate"],
                })


raw = pd.DataFrame(rows)

print()
print("SEED COUNTS")
print("=" * 80)

counts = (
    raw.groupby(
        ["environment", "algorithm", "speed_scale"]
    )["training_seed"]
    .nunique()
)

print(counts.to_string())

if (counts != 5).any():
    raise RuntimeError(
        "At least one condition does not contain all five seeds."
    )


summary = (
    raw.groupby(
        ["environment", "algorithm", "speed_scale"]
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
print("V9 VS V10 SPEED ROBUSTNESS @ 1M")
print("=" * 120)

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


wide = summary.pivot(
    index=["algorithm", "speed_scale"],
    columns="environment",
    values=[
        "success_mean",
        "success_std",
    ],
)

wide.columns = [
    "_".join(col)
    for col in wide.columns
]

wide = wide.reset_index()

wide["absolute_improvement"] = (
    wide["success_mean_v10"]
    - wide["success_mean_v9"]
)

wide["relative_improvement"] = (
    wide["success_mean_v10"]
    / wide["success_mean_v9"]
)


print()
print("DOMAIN-RANDOMIZATION EFFECT")
print("=" * 100)

print(
    wide[
        [
            "algorithm",
            "speed_scale",
            "success_mean_v9",
            "success_mean_v10",
            "absolute_improvement",
            "relative_improvement",
        ]
    ].to_string(
        index=False,
        formatters={
            "success_mean_v9":
                lambda x: f"{x:.1%}",

            "success_mean_v10":
                lambda x: f"{x:.1%}",

            "absolute_improvement":
                lambda x: f"{x:+.1%}",

            "relative_improvement":
                lambda x: f"{x:.2f}x",
        },
    )
)


raw_path = (
    OUTPUT_DIR
    / "v9_v10_speed_comparison_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "v9_v10_speed_comparison_summary.csv"
)

comparison_path = (
    OUTPUT_DIR
    / "v9_v10_speed_comparison_effect.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

wide.to_csv(
    comparison_path,
    index=False,
)

print()
print("Saved:")
print(f"  {raw_path}")
print(f"  {summary_path}")
print(f"  {comparison_path}")
