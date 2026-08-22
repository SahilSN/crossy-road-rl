from pathlib import Path

import pandas as pd


RUN_ROOT = Path("results/runs/v9")
OUTPUT_DIR = Path(
    "results/figures/composition_ood"
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
    "all_river",
    "river_heavy",
    "standard",
    "road_heavy",
    "all_road",
]

CONDITION_LABELS = {
    "all_river": "All river",
    "river_heavy": "1 road / 3 rivers",
    "standard": "Standard mixture",
    "road_heavy": "3 roads / 1 river",
    "all_road": "All road",
}


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"

    return f"{algorithm}_seed{seed}"


def eval_filename(condition):
    if condition == "standard":
        return "evaluation.csv"

    return (
        f"evaluation_composition_"
        f"{condition}.csv"
    )


rows = []

# ============================================================
# Load all five-seed 1M evaluations
# ============================================================

for algorithm in ALGORITHMS:
    for seed in range(5):

        run_dir = (
            RUN_ROOT
            / run_name(
                algorithm,
                seed,
            )
        )

        for condition in CONDITIONS:

            path = (
                run_dir
                / eval_filename(condition)
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing evaluation: {path}"
                )

            df = pd.read_csv(path)

            final = df[
                df["checkpoint_steps"]
                == 1_000_000
            ]

            if len(final) != 1:
                raise RuntimeError(
                    f"Expected exactly one "
                    f"1M row in {path}, "
                    f"found {len(final)}"
                )

            row = final.iloc[0]

            rows.append({
                "algorithm":
                    algorithm,

                "training_seed":
                    seed,

                "condition":
                    condition,

                "condition_label":
                    CONDITION_LABELS[
                        condition
                    ],

                "success_rate":
                    row["success_rate"],

                "avg_reward":
                    row["avg_reward"],

                "avg_max_row":
                    row["avg_max_row"],

                "road_collision_rate":
                    row[
                        "road_collision_rate"
                    ],

                "drowning_rate":
                    row[
                        "drowning_rate"
                    ],

                "timeout_rate":
                    row[
                        "timeout_rate"
                    ],
            })


raw = pd.DataFrame(rows)


# ============================================================
# Validate completeness
# ============================================================

counts = (
    raw.groupby(
        [
            "algorithm",
            "condition",
        ]
    )["training_seed"]
    .nunique()
)

print()
print("SEED COUNTS")
print("=" * 80)
print(counts.to_string())

if (counts != 5).any():
    raise RuntimeError(
        "At least one condition "
        "does not contain all five seeds."
    )


# ============================================================
# Aggregate
# ============================================================

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


# Preserve intended condition order.
summary["condition"] = pd.Categorical(
    summary["condition"],
    categories=CONDITIONS,
    ordered=True,
)

summary = summary.sort_values(
    [
        "algorithm",
        "condition",
    ]
)


# ============================================================
# Main summary
# ============================================================

print()
print(
    "V9 COMPOSITION GENERALIZATION @ 1M"
)
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


# ============================================================
# Compact success table
# ============================================================

success_table = (
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
print("SUCCESS-RATE LADDER")
print("=" * 90)

display = success_table.copy()

for column in display.columns:
    display[column] = (
        display[column]
        .map(
            lambda x: f"{x:.1%}"
        )
    )

print(display.to_string())


# ============================================================
# Extreme unseen-composition gap
# ============================================================

extreme = (
    success_table[
        "all_river"
    ]
    - success_table[
        "all_road"
    ]
)

print()
print(
    "ALL-RIVER MINUS ALL-ROAD SUCCESS GAP"
)
print("=" * 60)

for algorithm in ALGORITHMS:
    print(
        f"{algorithm:6s}: "
        f"{extreme[algorithm]:+.1%}"
    )


# ============================================================
# Save
# ============================================================

raw_path = (
    OUTPUT_DIR
    / "v9_composition_ood_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "v9_composition_ood_summary.csv"
)

ladder_path = (
    OUTPUT_DIR
    / "v9_composition_success_ladder.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

success_table.to_csv(
    ladder_path,
)

print()
print("Saved:")
print(f"  {raw_path}")
print(f"  {summary_path}")
print(f"  {ladder_path}")
