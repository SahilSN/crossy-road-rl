import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = "results/benchmark_all_runs.csv"
OUTPUT = Path("results/figures/cross_environment/v8_v9_failure_modes.png")

ENVIRONMENTS = ["v8", "v9"]
ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]

ALGORITHM_LABELS = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "dqn": "DQN",
    "qrdqn": "QR-DQN",
}

ENV_LABELS = {
    "v8": "v8 — Fixed Mixed Mechanics",
    "v9": "v9 — Procedural Mixed Mechanics",
}


def main():
    df = pd.read_csv(INPUT)

    final = df[
        (df["checkpoint_steps"] == 1_000_000)
        & (df["environment"].isin(ENVIRONMENTS))
        & (df["algorithm"].isin(ALGORITHMS))
    ].copy()

    summary = (
        final.groupby(["environment", "algorithm"])
        .agg(
            success=("success_rate", "mean"),
            road_collision=("road_collision_rate", "mean"),
            drowning=("drowning_rate", "mean"),
            timeout=("timeout_rate", "mean"),
            n_seeds=("training_seed", "nunique"),
        )
        .reset_index()
    )

    print()
    print("V8/V9 FAILURE-MODE SUMMARY @ 1M")
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

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    for ax, env in zip(axes, ENVIRONMENTS):
        env_data = (
            summary[
                summary["environment"] == env
            ]
            .set_index("algorithm")
            .reindex(ALGORITHMS)
        )

        success = env_data["success"] * 100
        road = env_data["road_collision"] * 100
        drowning = env_data["drowning"] * 100
        timeout = env_data["timeout"] * 100

        x = range(len(ALGORITHMS))

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

        ax.set_title(ENV_LABELS[env])

        ax.set_xticks(
            list(x),
            [
                ALGORITHM_LABELS[a]
                for a in ALGORITHMS
            ],
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Episode Outcome (%)")

        ax.grid(
            axis="y",
            alpha=0.25,
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
        "Failure-Mode Composition at 1M Training Steps",
        fontsize=15,
        y=1.03,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
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


if __name__ == "__main__":
    main()
