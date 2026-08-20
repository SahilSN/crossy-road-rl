from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env import CrossyRoadEnv


TOTAL_TIMESTEPS = 1_000_000

env = Monitor(CrossyRoadEnv())

checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="results/ppo_v3_checkpoints/",
    name_prefix="ppo_basic_v3",
)

model = PPO(
    "MlpPolicy",
    env,
    seed=0,
    verbose=1,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
)

model.save("results/ppo_basic_v3_1m")

print("Saved final PPO v2 model.")
