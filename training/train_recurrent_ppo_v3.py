from pathlib import Path
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env import CrossyRoadEnv


TOTAL_TIMESTEPS = 1_000_000
SEED = 0

run_dir = Path("results/runs/recurrent_ppo_seed0")
checkpoint_dir = run_dir / "checkpoints"

run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

env = Monitor(CrossyRoadEnv())

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    seed=SEED,
    verbose=1,
)

callback = CheckpointCallback(
    save_freq=200_000,
    save_path=str(checkpoint_dir),
    name_prefix="recurrent_ppo_seed0",
)

start = time.perf_counter()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback,
)

wall_time = time.perf_counter() - start

model.save(str(run_dir / "final_model"))

print()
print("=== Run complete ===")
print("Algorithm: recurrent_ppo")
print(f"Seed: {SEED}")
print(f"Timesteps: {TOTAL_TIMESTEPS}")
print(f"Wall time: {wall_time:.2f} seconds")
print(f"Output: {run_dir}")
