from pathlib import Path
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


SEED = 0

run_dir = Path(
    "results/runs/v6_local1_recurrent_tuning/"
    "recurrent_ppo_ent02_seed0"
)

checkpoint_dir = run_dir / "checkpoints"

source_model = (
    checkpoint_dir
    / "recurrent_ppo_ent02_seed0_400000_steps.zip"
)

if not source_model.exists():
    raise FileNotFoundError(
        f"Could not find {source_model}"
    )

env = Monitor(
    CrossyRoadEnvV6(
        observation_horizon=1
    ),
    filename=str(run_dir / "continue_monitor.csv"),
)

print("=" * 72)
print("Continuing Recurrent PPO ent02")
print("=" * 72)
print(f"Starting checkpoint: {source_model}")
print("Existing steps:      400000")
print("Additional steps:    600000")
print("Final total:         1000000")
print("=" * 72)

model = RecurrentPPO.load(
    str(source_model),
    env=env,
)

# Save every additional 200k steps:
#
# +200k -> total 600k
# +400k -> total 800k
# +600k -> total 1M
callback = CheckpointCallback(
    save_freq=200_000,
    save_path=str(checkpoint_dir),
    name_prefix="recurrent_ppo_ent02_continued",
)

start = time.perf_counter()

model.learn(
    total_timesteps=600_000,
    callback=callback,
    reset_num_timesteps=False,
)

wall_time = time.perf_counter() - start

model.save(
    str(run_dir / "final_model_1m")
)

env.close()

print()
print("=" * 72)
print("Continuation complete")
print("=" * 72)
print(f"Additional wall time: {wall_time:.2f} seconds")
print(f"Final model: {run_dir / 'final_model_1m.zip'}")
