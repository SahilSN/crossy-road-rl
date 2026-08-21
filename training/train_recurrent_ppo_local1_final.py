from pathlib import Path
import argparse
import csv
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=200_000,
    )

    args = parser.parse_args()

    run_name = f"recurrent_ppo_ent02_seed{args.seed}"

    run_dir = (
        Path("results/runs/v6_local1")
        / run_name
    )

    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        CrossyRoadEnvV6(
            observation_horizon=1
        ),
        filename=str(run_dir / "monitor.csv"),
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        ent_coef=0.02,
        gamma=0.99,
        seed=args.seed,
        verbose=1,
    )

    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
    )

    start = time.perf_counter()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
    )

    wall_time = time.perf_counter() - start

    model.save(str(run_dir / "final_model"))

    env.close()

    metadata = {
        "environment": "v6_local1",
        "algorithm": "recurrent_ppo",
        "config": "ent02",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "wall_time_seconds": wall_time,
        "model_path": str(run_dir / "final_model.zip"),
    }

    with (run_dir / "run_metadata.csv").open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=metadata.keys(),
        )
        writer.writeheader()
        writer.writerow(metadata)

    print()
    print("=" * 72)
    print("Run complete")
    print("=" * 72)

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
