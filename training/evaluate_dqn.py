from collections import Counter
from pathlib import Path

from stable_baselines3 import DQN

from crossyroad_rl.env import CrossyRoadEnv


NUM_EPISODES = 100

ACTION_NAMES = {
    0: "wait",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
}


MODELS = [
    ("200k", "results/dqn_checkpoints/dqn_basic_v1_200000_steps"),
    ("400k", "results/dqn_checkpoints/dqn_basic_v1_400000_steps"),
    ("600k", "results/dqn_checkpoints/dqn_basic_v1_600000_steps"),
    ("800k", "results/dqn_checkpoints/dqn_basic_v1_800000_steps"),
    ("1m", "results/dqn_basic_v1_1m"),
]


def evaluate(label, model_path):
    env = CrossyRoadEnv()
    model = DQN.load(model_path)

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

        episode_reward = 0.0
        episode_steps = 0

        while True:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

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

    total_actions = sum(action_counts.values())

    print()
    print("=" * 60)
    print(f"DQN checkpoint: {label}")
    print("=" * 60)

    print(f"Episodes: {NUM_EPISODES}")
    print(f"Successes: {successes}")
    print(f"Failures: {NUM_EPISODES - successes}")
    print(f"Success rate: {successes / NUM_EPISODES:.2%}")
    print(f"Average reward: {total_reward / NUM_EPISODES:.3f}")
    print(f"Average episode length: {total_steps / NUM_EPISODES:.2f}")
    print(f"Average max row: {sum(max_rows) / NUM_EPISODES:.2f}")

    print()
    print("Action distribution:")

    for action in range(5):
        count = action_counts[action]
        pct = count / total_actions if total_actions else 0.0

        print(
            f"  {ACTION_NAMES[action]:>8}: "
            f"{count:5d} ({pct:6.2%})"
        )

    print()
    print("Collision rows:")

    if failure_rows:
        for row in sorted(failure_rows):
            print(f"  row {row}: {failure_rows[row]}")
    else:
        print("  None")

    print()
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

    return {
        "checkpoint": label,
        "success_rate": successes / NUM_EPISODES,
        "avg_reward": total_reward / NUM_EPISODES,
        "avg_length": total_steps / NUM_EPISODES,
        "avg_max_row": sum(max_rows) / NUM_EPISODES,
    }


results = []

for label, path in MODELS:
    results.append(
        evaluate(label, path)
    )


print()
print("=" * 60)
print("DQN LEARNING CURVE SUMMARY")
print("=" * 60)

print(
    f"{'Checkpoint':<12}"
    f"{'Success':>10}"
    f"{'Reward':>10}"
    f"{'Length':>10}"
    f"{'Max row':>10}"
)

for result in results:
    print(
        f"{result['checkpoint']:<12}"
        f"{result['success_rate']:>9.1%}"
        f"{result['avg_reward']:>10.3f}"
        f"{result['avg_length']:>10.2f}"
        f"{result['avg_max_row']:>10.2f}"
    )
