from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from crossyroad_rl.env import CrossyRoadEnv


env = CrossyRoadEnv()
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0,
)

model.learn(
    total_timesteps=200_000
)

model.save("results/ppo_basic_v1")

print("Saved model to results/ppo_basic.zip")
