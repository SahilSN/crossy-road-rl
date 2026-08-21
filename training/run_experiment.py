import argparse
import csv
import time
from pathlib import Path

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN, TRPO

from crossyroad_rl.env import CrossyRoadEnv
from crossyroad_rl.env_v4 import CrossyRoadEnvV4
from crossyroad_rl.env_v5 import CrossyRoadEnvV5


ENVIRONMENTS = {
    "v3": CrossyRoadEnv,
    "v4": CrossyRoadEnvV4,
    "v5": CrossyRoadEnvV5,
}


ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
    "qrdqn": QRDQN,
    "a2c": A2C,
    "trpo": TRPO,
}


def build_model(algorithm, env, seed):
    if algorithm == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
        )

    if algorithm == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=7e-4,
            gamma=0.99,
            seed=seed,
            verbose=1,
        )

    if algorithm == "trpo":
        return TRPO(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            gamma=0.99,
            seed=seed,
            verbose=1,
        )

    if algorithm == "dqn":
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.20,
            seed=seed,
            verbose=1,
        )

    if algorithm == "qrdqn":
        return QRDQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_fraction=0.20,
            policy_kwargs={
                "n_quantiles": 50,
            },
            seed=seed,
            verbose=1,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["v3", "v4", "v5"],
        default="v3",
        help="Environment version to train on.",
    )

    parser.add_argument(
        "--algorithm",
        choices=["ppo", "dqn", "qrdqn", "a2c", "trpo"],
        required=True,
    )

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

    if args.algorithm == "qrdqn":
        run_name = f"qrdqn_50q_seed{args.seed}"
    else:
        run_name = f"{args.algorithm}_seed{args.seed}"

    if args.env == "v3":
        # Preserve the existing v3 directory layout.
        run_dir = Path("results/runs") / run_name
    else:
        run_dir = Path("results/runs") / args.env / run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_class = ENVIRONMENTS[args.env]
    env = env_class()

    env = Monitor(
        env,
        filename=str(run_dir / "monitor.csv"),
        info_keywords=("success", "collision", "max_y"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
    )

    model = build_model(
        args.algorithm,
        env,
        args.seed,
    )

    start_time = time.perf_counter()

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
    )

    wall_time = time.perf_counter() - start_time

    final_model = run_dir / "final_model"
    model.save(str(final_model))

    metadata_path = run_dir / "run_metadata.csv"

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "environment",
                "algorithm",
                "seed",
                "timesteps",
                "checkpoint_freq",
                "wall_time_seconds",
                "model_path",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "environment": args.env,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "checkpoint_freq": args.checkpoint_freq,
            "wall_time_seconds": wall_time,
            "model_path": str(final_model),
        })

    print()
    print("=== Run complete ===")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Seed: {args.seed}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Wall time: {wall_time:.2f} seconds")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
