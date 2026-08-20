from collections import Counter

from stable_baselines3 import PPO, DQN

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
    ("PPO", "200k", PPO, "results/ppo_v2_checkpoints/ppo_basic_v2_200000_steps"),
    ("PPO", "400k", PPO, "results/ppo_v2_checkpoints/ppo_basic_v2_400000_steps"),
    ("PPO", "600k", PPO, "results/ppo_v2_checkpoints/ppo_basic_v2_600000_steps"),
    ("PPO", "800k", PPO, "results/ppo_v2_checkpoints/ppo_basic_v2_800000_steps"),
    ("PPO", "1m",   PPO, "results/ppo_basic_v2_1m"),

    ("DQN", "200k", DQN, "results/dqn_v2_checkpoints/dqn_basic_v2_200000_steps"),
    ("DQN", "400k", DQN, "results/dqn_v2_checkpoints/dqn_basic_v2_400000_steps"),
    ("DQN", "600k", DQN, "results/dqn_v2_checkpoints/dqn_basic_v2_600000_steps"),
    ("DQN", "800k", DQN, "results/dqn_v2_checkpoints/dqn_basic_v2_800000_steps"),
    ("DQN", "1m",   DQN, "results/dqn_basic_v2_1m"),
]


def evaluate(algorithm, checkpoint, model_class, model_path):
    env = CrossyRoadEnv()
    model = model_class.load(model_path)

    successes = 0
    total_reward = 0.0
    total_steps = 0
    max_rows = []

    action_counts = Counter()
    collision_rows = Counter()

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
                collision_rows[env.player_y] += 1

    total_actions = sum(action_counts.values())

    result = {
        "algorithm": algorithm,
        "checkpoint": checkpoint,
        "success_rate": successes / NUM_EPISODES,
        "avg_reward": total_reward / NUM_EPISODES,
        "avg_length": total_steps / NUM_EPISODES,
        "avg_max_row": sum(max_rows) / NUM_EPISODES,
    }

    print()
    print("=" * 65)
    print(f"{algorithm} checkpoint: {checkpoint}")
    print("=" * 65)

    print(f"Success rate: {result['success_rate']:.2%}")
    print(f"Average reward: {result['avg_reward']:.3f}")
    print(f"Average episode length: {result['avg_length']:.2f}")
    print(f"Average max row: {result['avg_max_row']:.2f}")

    print()
    print("Action distribution:")

    for action in range(5):
        count = action_counts[action]
        pct = count / total_actions if total_actions else 0

        print(
            f"  {ACTION_NAMES[action]:>8}: "
            f"{count:5d} ({pct:6.2%})"
        )

    print()
    print("Collision rows:")

    if collision_rows:
        for row in sorted(collision_rows):
            print(f"  row {row}: {collision_rows[row]}")
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

    return result


results = []

for algorithm, checkpoint, model_class, model_path in MODELS:
    results.append(
        evaluate(
            algorithm,
            checkpoint,
            model_class,
            model_path,
        )
    )


print()
print("=" * 72)
print("V2 BENCHMARK SUMMARY")
print("=" * 72)

print(
    f"{'Algorithm':<10}"
    f"{'Steps':<10}"
    f"{'Success':>10}"
    f"{'Reward':>10}"
    f"{'Length':>10}"
    f"{'Max row':>10}"
)

for r in results:
    print(
        f"{r['algorithm']:<10}"
        f"{r['checkpoint']:<10}"
        f"{r['success_rate']:>9.1%}"
        f"{r['avg_reward']:>10.3f}"
        f"{r['avg_length']:>10.2f}"
        f"{r['avg_max_row']:>10.2f}"
    )
