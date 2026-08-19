from stable_baselines3 import PPO
from crossyroad_rl.env import CrossyRoadEnv

NUM_EPISODES = 100

env = CrossyRoadEnv()
model = PPO.load("results/ppo_basic")

successes = 0
collisions = 0
total_reward = 0.0
total_steps = 0
max_rows = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=1000 + episode)

    done = False
    episode_reward = 0.0
    episode_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        done = terminated or truncated

    total_reward += episode_reward
    total_steps += episode_steps
    max_rows.append(info["max_y"])

    if info["success"]:
        successes += 1
    else:
        collisions += 1

print(f"Episodes: {NUM_EPISODES}")
print(f"Successes: {successes}")
print(f"Failures: {collisions}")
print(f"Success rate: {successes / NUM_EPISODES:.2%}")
print(f"Average reward: {total_reward / NUM_EPISODES:.3f}")
print(f"Average episode length: {total_steps / NUM_EPISODES:.2f}")
print(f"Average max row: {sum(max_rows) / NUM_EPISODES:.2f}")
