from crossyroad_rl.env import CrossyRoadEnv

NUM_EPISODES = 20

env = CrossyRoadEnv()

successes = 0
collisions = 0

for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=episode)

    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    if info["success"]:
        successes += 1

    if info["collision"]:
        collisions += 1

    print(
        f"episode={episode:02d}",
        f"steps={steps:03d}",
        f"reward={total_reward:.2f}",
        f"max_y={info['max_y']}",
        f"success={info['success']}",
        f"collision={info['collision']}",
    )

print()
print(f"Random-policy successes: {successes}/{NUM_EPISODES}")
print(f"Random-policy collisions: {collisions}/{NUM_EPISODES}")
