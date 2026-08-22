from stable_baselines3.common.env_checker import check_env

from crossyroad_rl.env_v8 import CrossyRoadEnvV8


env = CrossyRoadEnvV8()

obs, info = env.reset(seed=0)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Road rows:", env.road_rows)
print("River rows:", env.river_rows)

print()
print("Roads:")
for row in env.road_rows:
    print(
        row,
        env.lanes[row],
    )

print()
print("Rivers:")
for row in env.river_rows:
    print(
        row,
        env.rivers[row],
    )

check_env(
    env,
    warn=True,
)

print()
print("Environment check passed.")
