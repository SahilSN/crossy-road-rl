import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = "results/benchmark_all_runs.csv"
OUTPUT_DIR = Path("results/figures/cross_environment")

ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

COMPARISONS = [
    {
        "env_a": "v4",
        "env_b": "v5",
        "label_a": "v4 — Global Observation",
        "label_b": "v5 — Local2 Observation",
        "title": "Observation Representation: Global vs Local",
        "filename": "v4_vs_v5_observation.png",
    },
    {
        "env_a": "v5",
        "env_b": "v7",
        "label_a": "v5 — Fixed Roads",
        "label_b": "v7 — Procedural Roads",
        "title": "Road Layout: Fixed vs Procedural",
        "filename": "v5_vs_v7_procedural_roads.png",
    },
    {
        "env_a": "v8",
        "env_b": "v9",
        "label_a": "v8 — Fixed Mixed Mechanics",
        "label_b": "v9 — Procedural Mixed Mechanics",
        "title": "Mixed Mechanics: Fixed vs Procedural",
        "filename": "v8_vs_v9_procedural_mixed.png",
    },
    {
        "env_a": "v7",
        "env_b": "v9",
        "label_a": "v7 — Procedural Roads",
        "label_b": "v9 — Procedural Mixed Mechanics",
        "title": "Procedural Environment: Roads vs Mixed Mechanics",
        "filename": "v7_vs_v9_procedural_mechanics.png",
    },
]


def aggregate(df):
    return (
        df.groupby(
            ["environment", "algorithm", "checkpoint_steps"]
        )
        .agg(
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            n_seeds=("training_seed", "nunique"),
        )
        .reset_index()
    )


def plot_environment(ax, data, env, title):
    env_data = data[data["environment"] == env]

    for algorithm in ALGORITHMS:
        sub = (
            env_data[
                env_data["algorithm"] == algorithm
            ]
            .sort_values("checkpoint_steps")
        )

        if sub.empty:
            continue

        x = sub["checkpoint_steps"] / 1000
        mean = sub["success_mean"] * 100
        std = sub["success_std"] * 100

        line, = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=ALGORITHM_LABELS[algorithm],
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.18,
            color=line.get_color(),
        )

    ax.set_title(title)
    ax.set_xlabel("Training Steps (thousands)")
    ax.set_ylabel("Success Rate (%)")

    ax.set_xlim(200, 1000)
    ax.set_ylim(0, 105)

    ax.set_xticks(
        [200, 400, 600, 800, 1000]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )


def main():
    df = pd.read_csv(INPUT)

    df = df[
        df["algorithm"].isin(ALGORITHMS)
    ].copy()

    data = aggregate(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for comparison in COMPARISONS:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharey=True,
        )

        plot_environment(
            axes[0],
            data,
            comparison["env_a"],
            comparison["label_a"],
        )

        plot_environment(
            axes[1],
            data,
            comparison["env_b"],
            comparison["label_b"],
        )

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
        )

        fig.suptitle(
            comparison["title"],
            fontsize=15,
            y=1.02,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.90]
        )

        output = (
            OUTPUT_DIR
            / comparison["filename"]
        )

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
