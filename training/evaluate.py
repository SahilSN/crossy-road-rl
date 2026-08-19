from collections import Counter

from stable_baselines3 import PPO
from crossyroad_rl.env import CrossyRoadEnv


NUM_EPISODES = 100

ACTION_NAMES = {
    0: "wait",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
}

env = CrossyRoadEnv()
model = PPO.load("results/ppo_basic_v1_1m")

successes = 0
collisions = 0

total_reward = 0.0
total_steps = 0

max_rows = []
action_counts = Counter()
failure_rows = Counter()

successful_steps = []
failed_steps = []


for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=1000 + episode)

    done = False
    episode_reward = 0.0
    episode_steps = 0

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        done = terminated or truncated

    total_reward += episode_reward
    total_steps += episode_steps
    max_rows.append(info["max_y"])

    if info["success"]:
        successes += 1
        successful_steps.append(episode_steps)

    else:
        failed_steps.append(episode_steps)

        if info.get("collision", False):
            collisions += 1
            failure_rows[env.player_y] += 1


print("=== Overall performance ===")
print(f"Episodes: {NUM_EPISODES}")
print(f"Successes: {successes}")
print(f"Failures: {NUM_EPISODES - successes}")
print(f"Success rate: {successes / NUM_EPISODES:.2%}")
print(f"Average reward: {total_reward / NUM_EPISODES:.3f}")
print(f"Average episode length: {total_steps / NUM_EPISODES:.2f}")
print(f"Average max row: {sum(max_rows) / NUM_EPISODES:.2f}")

print()
print("=== Action distribution ===")

total_actions = sum(action_counts.values())

for action in range(5):
    count = action_counts[action]
    percentage = count / total_actions if total_actions else 0

    print(
        f"{ACTION_NAMES[action]:>8}: "
        f"{count:5d} "
        f"({percentage:6.2%})"
    )

print()
print("=== Collision rows ===")

if failure_rows:
    for row in sorted(failure_rows):
        print(
            f"row {row}: "
            f"{failure_rows[row]} collisions"
        )
else:
    print("No collisions.")

print()
print("=== Episode lengths ===")

if successful_steps:
    print(
        "Average successful episode length: "
        f"{sum(successful_steps) / len(successful_steps):.2f}"
    )

if failed_steps:
    print(
        "Average failed episode length: "
        f"{sum(failed_steps) / len(failed_steps):.2f}"
    )
