from pathlib import Path
import argparse
import csv
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


CONFIGS = {
    "baseline": {
        "learning_rate": 3e-4,
        "ent_coef": 0.0,
        "lstm_hidden_size": 256,
        "n_epochs": 10,
    },
    "ent01": {
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
        "lstm_hidden_size": 256,
        "n_epochs": 10,
    },
    "lowlr_ent01": {
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
        "lstm_hidden_size": 256,
        "n_epochs": 10,
    },
    "ent02": {
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "lstm_hidden_size": 256,
        "n_epochs": 10,
    },

    # Runtime-optimized ent02 variants
    "ent02_lstm128": {
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "lstm_hidden_size": 128,
        "n_epochs": 10,
    },
    "ent02_lstm64": {
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "lstm_hidden_size": 64,
        "n_epochs": 10,
    },
    "ent02_lstm128_e5": {
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "lstm_hidden_size": 128,
        "n_epochs": 5,
    },
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        choices=CONFIGS.keys(),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=400_000,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=200_000,
    )

    args = parser.parse_args()

    cfg = CONFIGS[args.config]

    run_name = (
        f"recurrent_ppo_{args.config}_seed{args.seed}"
    )

    run_dir = (
        Path("results/runs/v6_local1_recurrent_tuning")
        / run_name
    )

    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = Monitor(
        CrossyRoadEnvV6(
            observation_horizon=1
        ),
        filename=str(run_dir / "monitor.csv"),
    )

    print("=" * 72)
    print("Recurrent PPO local1 tuning")
    print("=" * 72)
    print(f"Config:        {args.config}")
    print(f"Seed:          {args.seed}")
    print(f"Timesteps:     {args.timesteps}")
    print(f"Learning rate: {cfg['learning_rate']}")
    print(f"Entropy coef:  {cfg['ent_coef']}")
    print(f"LSTM hidden:   {cfg['lstm_hidden_size']}")
    print(f"PPO epochs:    {cfg['n_epochs']}")
    print("=" * 72)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        ent_coef=cfg["ent_coef"],
        gamma=0.99,
        n_epochs=cfg["n_epochs"],
        policy_kwargs={
            "lstm_hidden_size": cfg["lstm_hidden_size"],
        },
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

    final_path = run_dir / "final_model"
    model.save(str(final_path))

    env.close()

    metadata = {
        "environment": "v6_local1",
        "algorithm": "recurrent_ppo",
        "config": args.config,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": cfg["learning_rate"],
        "ent_coef": cfg["ent_coef"],
        "lstm_hidden_size": cfg["lstm_hidden_size"],
        "n_epochs": cfg["n_epochs"],
        "wall_time_seconds": wall_time,
        "model_path": str(final_path) + ".zip",
    }

    metadata_path = run_dir / "run_metadata.csv"

    with metadata_path.open(
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
