from crossyroad_rl.env import CrossyRoadEnv


env = CrossyRoadEnv()

obs, info = env.reset(seed=0)

print("Initial observation:", obs)

for i in range(100):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={i:02d}",
        f"action={action}",
        f"obs={obs}",
        f"reward={reward:.2f}",
        f"terminated={terminated}",
        f"info={info}",
    )

    if terminated or truncated:
        print("Episode ended.")
        break