from stable_baselines3.common.env_checker import check_env
from crossyroad_rl.env import CrossyRoadEnv

env = CrossyRoadEnv()

check_env(env)

print("Environment passed check_env.")
