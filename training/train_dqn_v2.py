from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env import CrossyRoadEnv


TOTAL_TIMESTEPS = 1_000_000

env = Monitor(CrossyRoadEnv())

checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="results/dqn_v2_checkpoints/",
    name_prefix="dqn_basic_v2",
)

model = DQN(
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
    seed=0,
    verbose=1,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
)

model.save("results/dqn_basic_v2_1m")

print("Saved final DQN v2 model.")
