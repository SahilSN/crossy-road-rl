import argparse
from pathlib import Path

import pandas as pd

from training.evaluate_run import evaluate_run
from crossyroad_rl.env_v9 import CrossyRoadEnvV9


ALGORITHMS = ["ppo", "trpo", "dqn", "qrdqn"]


def run_name_for(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"
    return f"{algorithm}_seed{seed}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speed-scale",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=1_000_000,
    )

    args = parser.parse_args()

    rows = []

    for algorithm in ALGORITHMS:
        for seed in range(5):
            run_name = run_name_for(
                algorithm,
                seed,
            )

            run_dir = (
                Path("results/runs/v9")
                / run_name
            )

            print()
            print("=" * 70)
            print(
                f"{algorithm.upper()} | "
                f"seed={seed} | "
                f"speed_scale={args.speed_scale}"
            )
            print("=" * 70)

            result = evaluate_run(
                env_name="v9",
                algorithm=algorithm,
                seed=seed,
                episodes=args.episodes,
                eval_seed_start=args.eval_seed_start,
                checkpoint_steps=args.checkpoint_steps,
                env_factory=lambda: CrossyRoadEnvV9(
                    speed_scale=args.speed_scale
                ),
            )

            result["speed_scale"] = args.speed_scale
            result["training_seed"] = seed
            result["algorithm"] = algorithm

            rows.append(result)

    df = pd.DataFrame(rows)

    output_dir = Path(
        "results/distribution_shift/v9_speed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = str(args.speed_scale).replace(".", "p")

    output = (
        output_dir
        / f"speed_{tag}.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
