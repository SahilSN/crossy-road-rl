from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path(
    "results/figures/composition_ood/"
    "v9_composition_ood_summary.csv"
)

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

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

CONDITIONS = [
    "all_river",
    "river_heavy",
    "standard",
    "road_heavy",
    "all_road",
]

X_LABELS = [
    "All river\n0R / 4V",
    "1R / 3V",
    "Standard\nmixture",
    "3R / 1V",
    "All road\n4R / 0V",
]


# ------------------------------------------------------------
# Load summary
# ------------------------------------------------------------

df = pd.read_csv(
    SUMMARY_PATH
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5.8)
)

x = np.arange(
    len(CONDITIONS)
)


for algorithm in ALGORITHMS:
    means = []
    stds = []

    for condition in CONDITIONS:
        row = df[
            (df["algorithm"] == algorithm)
            & (df["condition"] == condition)
        ]

        if len(row) != 1:
            raise RuntimeError(
                f"Expected one row for "
                f"{algorithm}/{condition}, "
                f"found {len(row)}"
            )

        row = row.iloc[0]

        means.append(
            row["success_mean"] * 100
        )

        stds.append(
            row["success_std"] * 100
        )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label=ALGORITHM_LABELS[
            algorithm
        ],
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "V9 Generalization Across Hazard Composition"
)

ax.set_xlabel(
    "Evaluation Composition"
)

ax.set_ylabel(
    "Success Rate (%)"
)

ax.set_xticks(
    x,
    X_LABELS,
)

ax.set_ylim(
    0,
    90,
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
# Save
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "v9_composition_ood_ladder.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved: {output_path}"
)
