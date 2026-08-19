from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from crossyroad_rl.env import CrossyRoadEnv

env = Monitor(CrossyRoadEnv())

model = PPO.load(
    "results/ppo_basic_v1_800k",
    env=env,
)

model.learn(
    total_timesteps=200_000,
    reset_num_timesteps=False,
)

model.save("results/ppo_basic_v1_1m")

print("Saved model to results/ppo_basic_v1_1m.zip")
