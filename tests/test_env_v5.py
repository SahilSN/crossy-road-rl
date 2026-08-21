from stable_baselines3.common.env_checker import check_env

from crossyroad_rl.env_v5 import CrossyRoadEnvV5


env = CrossyRoadEnvV5()

print("Observation shape:", env.observation_space.shape)
print("Action space:", env.action_space)

check_env(
    env,
    warn=True,
)

print("Environment check passed.")
