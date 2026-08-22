from stable_baselines3.common.env_checker import check_env

from crossyroad_rl.env_v9 import CrossyRoadEnvV9


env = CrossyRoadEnvV9()

obs, info = env.reset(seed=0)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

print("Hazard rows:", env.hazard_rows)
print("Hazard types:", env.hazard_types)

print("Road rows:", env.road_rows)
print("River rows:", env.river_rows)

check_env(
    env,
    warn=True,
)

print()
print("Environment check passed.")
