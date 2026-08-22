from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path(
    "results/figures/domain_randomization/"
    "v9_v10_speed_comparison_summary.csv"
)

OUTPUT_DIR = Path(
    "results/figures/domain_randomization"
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

ENV_LABELS = {
    "v9": "v9: fixed-speed training",
    "v10": "v10: speed-randomized training",
}

SPEEDS = [0.8, 1.0, 1.2, 1.4]


# ------------------------------------------------------------
# Load summary
# ------------------------------------------------------------

df = pd.read_csv(
    SUMMARY_PATH
)


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

expected = (
    len(ALGORITHMS)
    * 2
    * len(SPEEDS)
)

if len(df) != expected:
    raise RuntimeError(
        f"Expected {expected} summary rows, "
        f"found {len(df)}"
    )


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 8),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()


for ax, algorithm in zip(
    axes,
    ALGORITHMS,
):
    for env_name in ["v9", "v10"]:
        subset = df[
            (df["algorithm"] == algorithm)
            & (df["environment"] == env_name)
        ].sort_values(
            "speed_scale"
        )

        x = subset[
            "speed_scale"
        ].to_numpy()

        y = (
            subset[
                "success_mean"
            ].to_numpy()
            * 100
        )

        yerr = (
            subset[
                "success_std"
            ].to_numpy()
            * 100
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2,
            label=ENV_LABELS[
                env_name
            ],
        )

    ax.set_title(
        ALGORITHM_LABELS[
            algorithm
        ]
    )

    ax.set_xticks(
        SPEEDS
    )

    ax.set_xlim(
        0.75,
        1.45,
    )

    ax.set_ylim(
        0,
        70,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )


# ------------------------------------------------------------
# Shared labels
# ------------------------------------------------------------

fig.supxlabel(
    "Evaluation Hazard Speed Scale"
)

fig.supylabel(
    "Success Rate (%)"
)

fig.suptitle(
    "Effect of Speed Domain Randomization on V9 Robustness",
    fontsize=14,
)


# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.94,
    ),
    ncol=2,
    frameon=False,
)


fig.tight_layout(
    rect=[
        0.03,
        0.03,
        1.0,
        0.88,
    ]
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "v9_v10_speed_domain_randomization.png"
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
