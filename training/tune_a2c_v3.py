from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from crossyroad_rl.env import CrossyRoadEnv


CONFIGS = {
    "a2c_lr3e4_ent001": {
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
    },

    "a2c_lr1e4_ent001": {
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
    },

    "a2c_lr3e4_ent002": {
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
    },
}


for name, config in CONFIGS.items():

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    run_dir = Path("results/a2c_tuning") / name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(CrossyRoadEnv())

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        ent_coef=config["ent_coef"],
        gamma=0.99,
        seed=0,
        verbose=1,
    )

    callback = CheckpointCallback(
        save_freq=200_000,
        save_path=str(checkpoint_dir),
        name_prefix=name,
    )

    model.learn(
        total_timesteps=200_000,
        callback=callback,
    )

    model.save(str(run_dir / "final_model"))

    env.close()

    print(f"Saved {name}")
