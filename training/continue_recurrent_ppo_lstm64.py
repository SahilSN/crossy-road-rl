from pathlib import Path
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


run_dir = Path(
    "results/runs/v6_local1_recurrent_tuning/"
    "recurrent_ppo_ent02_lstm64_seed0"
)

checkpoint_dir = run_dir / "checkpoints"

source_model = (
    checkpoint_dir
    / "recurrent_ppo_ent02_lstm64_seed0_200000_steps.zip"
)

if not source_model.exists():
    raise FileNotFoundError(source_model)

env = Monitor(
    CrossyRoadEnvV6(observation_horizon=1),
    filename=str(run_dir / "continue_monitor.csv"),
)

model = RecurrentPPO.load(
    str(source_model),
    env=env,
)

callback = CheckpointCallback(
    save_freq=200_000,
    save_path=str(checkpoint_dir),
    name_prefix="recurrent_ppo_ent02_lstm64_continued",
)

print("=" * 72)
print("Continuing ent02_lstm64: 200k -> 400k")
print("=" * 72)

start = time.perf_counter()

model.learn(
    total_timesteps=200_000,
    callback=callback,
    reset_num_timesteps=False,
)

wall_time = time.perf_counter() - start

model.save(
    str(run_dir / "final_model_400k")
)

env.close()

print()
print(f"Additional wall time: {wall_time:.2f} sec")
print(f"Additional wall time: {wall_time / 60:.2f} min")
print("Finished at 400k total steps.")
