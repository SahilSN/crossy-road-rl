from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

SPEEDS = [0.8, 1.0, 1.2, 1.4]


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"

    return f"{algorithm}_seed{seed}"


def evaluation_filename(speed):
    if speed == 1.0:
        return "evaluation.csv"

    tag = str(speed).replace(".", "p")
    return f"evaluation_speed_{tag}.csv"


rows = []

# ------------------------------------------------------------
# Load final 1M evaluation from every seed / speed condition
# ------------------------------------------------------------

for algorithm in ALGORITHMS:
    for seed in range(5):
        run_dir = RUN_ROOT / run_name(
            algorithm,
            seed,
        )

        for speed in SPEEDS:
            path = (
                run_dir
                / evaluation_filename(speed)
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
                "speed_scale": speed,
                "success_rate": row["success_rate"],
                "avg_reward": row["avg_reward"],
                "avg_max_row": row["avg_max_row"],
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
print("SPEED-SHIFT EVALUATION COUNT")
print("=" * 70)

counts = raw.groupby(
    ["algorithm", "speed_scale"]
)["training_seed"].nunique()

print(counts.unstack().to_string())

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
        ["algorithm", "speed_scale"]
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


# ------------------------------------------------------------
# Print robustness table
# ------------------------------------------------------------

print()
print("V9 SPEED-SHIFT ROBUSTNESS @ 1M")
print("=" * 90)

display = summary.copy()

print(
    display.to_string(
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
# Plot robustness curve
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

for algorithm in ALGORITHMS:
    sub = (
        summary[
            summary["algorithm"] == algorithm
        ]
        .sort_values("speed_scale")
    )

    x = sub["speed_scale"]
    mean = sub["success_mean"] * 100
    std = sub["success_std"] * 100

    line, = ax.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=ALGORITHM_LABELS[algorithm],
    )

    lower = (mean - std).clip(lower=0)
    upper = (mean + std).clip(upper=100)

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=0.12,
        color=line.get_color(),
    )


# Mark training distribution
ax.axvline(
    1.0,
    linestyle="--",
    alpha=0.5,
)

ax.text(
    1.01,
    98,
    "Training distribution",
    rotation=90,
    va="top",
    fontsize=9,
)

ax.set_title(
    "V9 Robustness to Evaluation-Time Hazard Speed Shift"
)

ax.set_xlabel(
    "Hazard Speed Scale"
)

ax.set_ylabel(
    "Success Rate (%)"
)

ax.set_xticks(SPEEDS)

ax.set_ylim(0, 100)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

plot_path = (
    OUTPUT_DIR
    / "v9_speed_shift_robustness.png"
)

fig.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# Save data
# ------------------------------------------------------------

raw_path = (
    OUTPUT_DIR
    / "v9_speed_shift_raw.csv"
)

summary_path = (
    OUTPUT_DIR
    / "v9_speed_shift_summary.csv"
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
print(f"  {plot_path}")
print(f"  {raw_path}")
print(f"  {summary_path}")
