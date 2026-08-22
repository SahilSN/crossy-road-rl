from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT = "results/figures/distribution_shift/v9_speed_shift_raw.csv"
OUTPUT = Path(
    "results/figures/distribution_shift/"
    "v9_speed_shift_failure_modes.png"
)

ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

SPEEDS = [0.8, 1.0, 1.2, 1.4]


df = pd.read_csv(INPUT)

summary = (
    df.groupby(["algorithm", "speed_scale"])
    .agg(
        success=("success_rate", "mean"),
        road_collision=("road_collision_rate", "mean"),
        drowning=("drowning_rate", "mean"),
        timeout=("timeout_rate", "mean"),
    )
    .reset_index()
)

print()
print("V9 SPEED-SHIFT FAILURE MODES @ 1M")
print("=" * 100)

display = summary.copy()

for col in [
    "success",
    "road_collision",
    "drowning",
    "timeout",
]:
    display[col] *= 100

print(
    display.to_string(
        index=False,
        formatters={
            "success": lambda x: f"{x:.1f}%",
            "road_collision": lambda x: f"{x:.1f}%",
            "drowning": lambda x: f"{x:.1f}%",
            "timeout": lambda x: f"{x:.1f}%",
        },
    )
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11, 8),
    sharey=True,
)

axes = axes.flatten()

for ax, algorithm in zip(
    axes,
    ALGORITHMS,
):
    sub = (
        summary[
            summary["algorithm"] == algorithm
        ]
        .set_index("speed_scale")
        .reindex(SPEEDS)
    )

    success = sub["success"] * 100
    road = sub["road_collision"] * 100
    drowning = sub["drowning"] * 100
    timeout = sub["timeout"] * 100

    x = range(len(SPEEDS))

    ax.bar(
        x,
        success,
        label="Success",
    )

    ax.bar(
        x,
        road,
        bottom=success,
        label="Road collision",
    )

    ax.bar(
        x,
        drowning,
        bottom=success + road,
        label="Drowning",
    )

    ax.bar(
        x,
        timeout,
        bottom=success + road + drowning,
        label="Timeout",
    )

    ax.set_title(
        ALGORITHM_LABELS[algorithm]
    )

    ax.set_xticks(
        list(x),
        [f"{speed:.1f}×" for speed in SPEEDS],
    )

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

for ax in axes[2:]:
    ax.set_xlabel(
        "Hazard Speed Scale"
    )

for ax in [
    axes[0],
    axes[2],
]:
    ax.set_ylabel(
        "Episode Outcome (%)"
    )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.97),
)

fig.suptitle(
    "V9 Failure Modes Under Evaluation-Time Speed Shift",
    fontsize=15,
    y=1.00,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print(f"Saved: {OUTPUT}")
