from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


TRANSFER_PATH = Path(
    "results/cross_env_transfer/cross_env_transfer_summary.csv"
)

BENCHMARK_PATH = Path(
    "results/benchmark_all_runs.csv"
)

OUTPUT_DIR = Path(
    "results/figures/cross_env_transfer"
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


# ------------------------------------------------------------
# Load transfer results
# ------------------------------------------------------------

transfer = pd.read_csv(
    TRANSFER_PATH
)


# ------------------------------------------------------------
# Load native 1M baselines
# ------------------------------------------------------------

benchmark = pd.read_csv(
    BENCHMARK_PATH
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
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def get_native(env_name, algorithm):
    row = native_summary[
        (native_summary["environment"] == env_name)
        & (native_summary["algorithm"] == algorithm)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one native row for "
            f"{env_name}/{algorithm}, "
            f"found {len(row)}"
        )

    row = row.iloc[0]

    return (
        row["success_mean"] * 100,
        row["success_std"] * 100,
    )


def get_transfer(train_env, eval_env, algorithm):
    row = transfer[
        (transfer["train_env"] == train_env)
        & (transfer["eval_env"] == eval_env)
        & (transfer["algorithm"] == algorithm)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected one transfer row for "
            f"{train_env}->{eval_env}/{algorithm}, "
            f"found {len(row)}"
        )

    row = row.iloc[0]

    return (
        row["success_mean"] * 100,
        row["success_std"] * 100,
    )


# ------------------------------------------------------------
# Build plotting data
# ------------------------------------------------------------

x = np.arange(
    len(ALGORITHMS)
)

bar_width = 0.34


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.5),
    sharey=True,
)


# ============================================================
# Panel 1: Train on v8
# ============================================================

native_means = []
native_stds = []

transfer_means = []
transfer_stds = []

for algorithm in ALGORITHMS:
    mean, std = get_native(
        "v8",
        algorithm,
    )

    native_means.append(mean)
    native_stds.append(std)

    mean, std = get_transfer(
        "v8",
        "v9",
        algorithm,
    )

    transfer_means.append(mean)
    transfer_stds.append(std)


ax = axes[0]

ax.bar(
    x - bar_width / 2,
    native_means,
    width=bar_width,
    yerr=native_stds,
    capsize=4,
    label="Evaluate on v8",
)

ax.bar(
    x + bar_width / 2,
    transfer_means,
    width=bar_width,
    yerr=transfer_stds,
    capsize=4,
    label="Evaluate on v9",
)

ax.set_title(
    "Train on Fixed v8"
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
    110,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
)


# ============================================================
# Panel 2: Train on v9
# ============================================================

native_means = []
native_stds = []

transfer_means = []
transfer_stds = []

for algorithm in ALGORITHMS:
    mean, std = get_native(
        "v9",
        algorithm,
    )

    native_means.append(mean)
    native_stds.append(std)

    mean, std = get_transfer(
        "v9",
        "v8",
        algorithm,
    )

    transfer_means.append(mean)
    transfer_stds.append(std)


ax = axes[1]

ax.bar(
    x - bar_width / 2,
    native_means,
    width=bar_width,
    yerr=native_stds,
    capsize=4,
    label="Evaluate on v9",
)

ax.bar(
    x + bar_width / 2,
    transfer_means,
    width=bar_width,
    yerr=transfer_stds,
    capsize=4,
    label="Evaluate on v8",
)

ax.set_title(
    "Train on Procedural v9"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_xticks(
    x,
    [
        ALGORITHM_LABELS[a]
        for a in ALGORITHMS
    ],
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
)


# ------------------------------------------------------------
# Overall title
# ------------------------------------------------------------

fig.suptitle(
    "Cross-Environment Transfer Between Fixed and Procedural Mixed-Mechanics Tasks",
    fontsize=13,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "v8_v9_cross_environment_transfer.png"
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
