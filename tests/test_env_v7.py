from stable_baselines3.common.env_checker import check_env

from crossyroad_rl.env_v7 import CrossyRoadEnvV7


env = CrossyRoadEnvV7()

obs, info = env.reset(seed=0)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Seed 0 road rows:", env.road_rows)

check_env(
    env,
    warn=True,
)

print("Environment check passed.")
