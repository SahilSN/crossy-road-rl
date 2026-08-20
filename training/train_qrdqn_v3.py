from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from crossyroad_rl.env import CrossyRoadEnv


TOTAL_TIMESTEPS = 1_000_000

env = Monitor(CrossyRoadEnv())

checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="results/qrdqn_v3_checkpoints/",
    name_prefix="qrdqn_basic_v3",
)

model = QRDQN(
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

model.save("results/qrdqn_basic_v3_1m")

print("Saved final QR-DQN v3 model.")
