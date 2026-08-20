from collections import Counter

from crossyroad_rl.env_v4 import CrossyRoadEnvV4


NUM_EPISODES = 100

env = CrossyRoadEnvV4()

successes = 0
collisions = 0
timeouts = 0

max_rows = []
collision_rows = Counter()
episode_lengths = []


for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=1000 + episode)

    steps = 0

    while True:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        steps += 1

        if terminated or truncated:
            break

    episode_lengths.append(steps)
    max_rows.append(info["max_y"])

    if info["success"]:
        successes += 1

    elif info["collision"]:
        collisions += 1
        collision_rows[env.player_y] += 1

    elif truncated:
        timeouts += 1


print()
print("=== Random-policy v4 sanity check ===")
print(f"Episodes: {NUM_EPISODES}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts: {timeouts}")

print(
    "Average max row:",
    sum(max_rows) / NUM_EPISODES,
)

print(
    "Average episode length:",
    sum(episode_lengths) / NUM_EPISODES,
)

print()
print("Collision rows:")

for row in sorted(collision_rows):
    print(
        f"  row {row}: "
        f"{collision_rows[row]}"
    )
