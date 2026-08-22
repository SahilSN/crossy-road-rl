from pathlib import Path

import pandas as pd


RUN_ROOT = Path("results/runs/v9")
OUTPUT_DIR = Path(
    "results/figures/layout_ood"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALGORITHMS = [
    "ppo",
    "trpo",
    "dqn",
    "qrdqn",
]

CONDITIONS = [
    "standard",
    "separated",
    "clustered",
]

LABELS = {
    "standard": "Standard",
    "separated": "Separated",
    "clustered": "Clustered OOD",
}


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"
    return f"{algorithm}_seed{seed}"


def eval_filename(condition):
    if condition == "standard":
        return "evaluation.csv"

    return f"evaluation_layout_{condition}.csv"


rows = []

for algorithm in ALGORITHMS:
    for seed in range(5):
        run_dir = (
            RUN_ROOT
            / run_name(algorithm, seed)
        )

        for condition in CONDITIONS:
            path = (
                run_dir
                / eval_filename(condition)
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}"
                )

            df = pd.read_csv(path)

            final = df[
                df["checkpoint_steps"]
                == 1_000_000
            ]

            if len(final) != 1:
                raise RuntimeError(
                    f"Expected one 1M row "
                    f"in {path}; found "
                    f"{len(final)}"
                )

            row = final.iloc[0]

            rows.append({
                "algorithm": algorithm,
                "training_seed": seed,
                "condition": condition,
                "condition_label":
                    LABELS[condition],
                "success_rate":
                    row["success_rate"],
                "avg_reward":
                    row["avg_reward"],
                "avg_max_row":
                    row["avg_max_row"],
                "road_collision_rate":
                    row["road_collision_rate"],
                "drowning_rate":
                    row["drowning_rate"],
                "timeout_rate":
                    row["timeout_rate"],
            })


raw = pd.DataFrame(rows)

counts = (
    raw.groupby(
        ["algorithm", "condition"]
    )["training_seed"]
    .nunique()
)

print()
print("SEED COUNTS")
print("=" * 80)
print(counts.to_string())

if (counts != 5).any():
    raise RuntimeError(
        "Missing one or more seeds."
    )


summary = (
    raw.groupby(
        [
            "algorithm",
            "condition",
            "condition_label",
        ],
        sort=False,
    )
    .agg(
        n_seeds=(
            "training_seed",
            "nunique",
        ),
        success_mean=(
            "success_rate",
            "mean",
        ),
        success_std=(
            "success_rate",
            "std",
        ),
        reward_mean=(
            "avg_reward",
            "mean",
        ),
        reward_std=(
            "avg_reward",
            "std",
        ),
        max_row_mean=(
            "avg_max_row",
            "mean",
        ),
        max_row_std=(
            "avg_max_row",
            "std",
        ),
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

summary["condition"] = pd.Categorical(
    summary["condition"],
    categories=CONDITIONS,
    ordered=True,
)

summary = summary.sort_values(
    ["algorithm", "condition"]
)


print()
print("V9 LAYOUT GENERALIZATION @ 1M")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        formatters={
            "success_mean":
                lambda x: f"{x:.1%}",
            "success_std":
                lambda x: f"{x:.1%}",
            "reward_mean":
                lambda x: f"{x:.3f}",
            "reward_std":
                lambda x: f"{x:.3f}",
            "max_row_mean":
                lambda x: f"{x:.2f}",
            "max_row_std":
                lambda x: f"{x:.2f}",
            "road_collision_mean":
                lambda x: f"{x:.1%}",
            "drowning_mean":
                lambda x: f"{x:.1%}",
            "timeout_mean":
                lambda x: f"{x:.1%}",
        },
    )
)


success = (
    summary.pivot(
        index="algorithm",
        columns="condition",
        values="success_mean",
    )
    .reindex(
        index=ALGORITHMS,
        columns=CONDITIONS,
    )
)

print()
print("SUCCESS-RATE COMPARISON")
print("=" * 70)

display = success.copy()

for col in display.columns:
    display[col] = display[col].map(
        lambda x: f"{x:.1%}"
    )

print(display.to_string())


print()
print("SHIFT RELATIVE TO STANDARD")
print("=" * 70)

for algorithm in ALGORITHMS:
    standard = success.loc[
        algorithm, "standard"
    ]

    separated = success.loc[
        algorithm, "separated"
    ]

    clustered = success.loc[
        algorithm, "clustered"
    ]

    print(
        f"{algorithm:6s} | "
        f"separated "
        f"{separated-standard:+.1%} | "
        f"clustered "
        f"{clustered-standard:+.1%}"
    )


raw_path = (
    OUTPUT_DIR
    / "v9_layout_ood_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "v9_layout_ood_summary.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

print()
print("Saved:")
print(f"  {raw_path}")
print(f"  {summary_path}")
