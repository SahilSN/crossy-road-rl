from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_PATH = Path(
    "results/figures/layout_ood/"
    "v9_layout_ood_summary.csv"
)

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

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

CONDITIONS = [
    "standard",
    "separated",
    "clustered",
]

CONDITION_LABELS = {
    "standard": "Standard",
    "separated": "Separated",
    "clustered": "Clustered OOD",
}


df = pd.read_csv(SUMMARY_PATH)


# ------------------------------------------------------------
# Collect data
# ------------------------------------------------------------

means = {}
stds = {}

for condition in CONDITIONS:
    means[condition] = []
    stds[condition] = []

    for algorithm in ALGORITHMS:
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

        means[condition].append(
            row["success_mean"] * 100
        )

        stds[condition].append(
            row["success_std"] * 100
        )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5.8)
)

x = np.arange(
    len(ALGORITHMS)
)

width = 0.24

offsets = {
    "standard": -width,
    "separated": 0.0,
    "clustered": width,
}


for condition in CONDITIONS:
    ax.bar(
        x + offsets[condition],
        means[condition],
        width,
        yerr=stds[condition],
        capsize=4,
        label=CONDITION_LABELS[
            condition
        ],
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_title(
    "V9 Generalization Across Spatial Layout Structure"
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
# Save
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "v9_layout_ood_success.png"
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
