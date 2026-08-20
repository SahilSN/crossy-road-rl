from pathlib import Path
import time

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env import CrossyRoadEnv


CONFIGS = {
    "a2c_n20": 20,
    "a2c_n50": 50,
    "a2c_n100": 100,
}

TOTAL_TIMESTEPS = 200_000
SEED = 0


for name, n_steps in CONFIGS.items():
    print()
    print("=" * 60)
    print(name)
    print(f"n_steps = {n_steps}")
    print("=" * 60)

    run_dir = Path("results/a2c_nsteps_tuning") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(CrossyRoadEnv())

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=n_steps,
        gamma=0.99,
        seed=SEED,
        verbose=1,
    )

    start = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
    )

    wall_time = time.perf_counter() - start

    model.save(str(run_dir / "final_model"))

    env.close()

    with open(run_dir / "metadata.txt", "w") as f:
        f.write(f"name={name}\n")
        f.write(f"seed={SEED}\n")
        f.write(f"timesteps={TOTAL_TIMESTEPS}\n")
        f.write("learning_rate=0.0003\n")
        f.write("ent_coef=0.01\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"wall_time_seconds={wall_time}\n")

    print()
    print(f"Finished {name}")
    print(f"Wall time: {wall_time:.2f} seconds")
    print(f"Saved to {run_dir}")


print()
print("All A2C n_steps tuning runs complete.")
