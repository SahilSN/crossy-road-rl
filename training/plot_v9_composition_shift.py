from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path("results/runs/v9")
OUTPUT_DIR = Path("results/figures/distribution_shift")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

CONDITIONS = [
    "standard",
    "road_heavy",
    "river_heavy",
]

CONDITION_LABELS = {
    "standard": "Standard",
    "road_heavy": "Road-heavy",
    "river_heavy": "River-heavy",
}


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"

    return f"{algorithm}_seed{seed}"


def evaluation_filename(condition):
    if condition == "standard":
        return "evaluation.csv"

    return (
        f"evaluation_composition_{condition}.csv"
    )


rows = []

# ------------------------------------------------------------
# Load 1M results
# ------------------------------------------------------------

for algorithm in ALGORITHMS:
    for seed in range(5):
        run_dir = (
            RUN_ROOT
            / run_name(algorithm, seed)
        )

        for condition in CONDITIONS:
            path = (
                run_dir
                / evaluation_filename(condition)
            )

            if not path.exists():
                print(f"Missing: {path}")
                continue

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
                "algorithm": algorithm,
                "training_seed": seed,
                "condition": condition,

                "success_rate": row[
                    "success_rate"
                ],

                "avg_reward": row[
                    "avg_reward"
                ],

                "avg_max_row": row[
                    "avg_max_row"
                ],

                "road_collision_rate": row[
                    "road_collision_rate"
                ],

                "drowning_rate": row[
                    "drowning_rate"
                ],

                "timeout_rate": row[
                    "timeout_rate"
                ],
            })


raw = pd.DataFrame(rows)

# ------------------------------------------------------------
# Validate completeness
# ------------------------------------------------------------

print()
print("COMPOSITION-SHIFT EVALUATION COUNT")
print("=" * 70)

counts = (
    raw.groupby(
        ["algorithm", "condition"]
    )["training_seed"]
    .nunique()
)

print(
    counts.unstack().to_string()
)

if (counts != 5).any():
    print()
    print(
        "WARNING: At least one condition does "
        "not contain all five seeds."
    )


# ------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------

summary = (
    raw.groupby(
        ["algorithm", "condition"]
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

        max_row_mean=(
            "avg_max_row",
            "mean",
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


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print()
print("V9 COMPOSITION-SHIFT ROBUSTNESS @ 1M")
print("=" * 105)

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

            "max_row_mean":
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


# ------------------------------------------------------------
# Grouped bar chart
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

x = np.arange(
    len(ALGORITHMS)
)

bar_width = 0.24

offsets = [
    -bar_width,
    0.0,
    bar_width,
]


for offset, condition in zip(
    offsets,
    CONDITIONS,
):
    means = []
    stds = []

    for algorithm in ALGORITHMS:
        row = summary[
            (summary["algorithm"] == algorithm)
            & (summary["condition"] == condition)
        ].iloc[0]

        means.append(
            row["success_mean"] * 100
        )

        stds.append(
            row["success_std"] * 100
        )

    ax.bar(
        x + offset,
        means,
        width=bar_width,
        yerr=stds,
        capsize=4,
        label=CONDITION_LABELS[
            condition
        ],
    )


ax.set_title(
    "V9 Robustness to Hazard-Composition Shift"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Success Rate (%)"
)

ax.set_xticks(
    x,
    [
        ALGORITHM_LABELS[a]
        for a in ALGORITHMS
    ],
)

ax.set_ylim(
    0,
    75,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

plot_path = (
    OUTPUT_DIR
    / "v9_composition_shift_robustness.png"
)

raw_path = (
    OUTPUT_DIR
    / "v9_composition_shift_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "v9_composition_shift_summary.csv"
)

fig.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

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
print(f"  {plot_path}")
print(f"  {raw_path}")
print(f"  {summary_path}")
