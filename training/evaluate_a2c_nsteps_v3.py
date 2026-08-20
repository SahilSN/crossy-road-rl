from collections import Counter
from pathlib import Path

from stable_baselines3 import A2C

from crossyroad_rl.env import CrossyRoadEnv


NUM_EPISODES = 100
EVAL_SEED_START = 1000

CONFIGS = [
    (
        "previous-best",
        "results/a2c_tuning/a2c_lr3e4_ent001/final_model",
    ),
    (
        "n_steps=20",
        "results/a2c_nsteps_tuning/a2c_n20/final_model",
    ),
    (
        "n_steps=50",
        "results/a2c_nsteps_tuning/a2c_n50/final_model",
    ),
    (
        "n_steps=100",
        "results/a2c_nsteps_tuning/a2c_n100/final_model",
    ),
]


ACTION_NAMES = {
    0: "wait",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
}


def evaluate(name, model_path):
    env = CrossyRoadEnv()
    model = A2C.load(model_path)

    successes = 0
    collisions = 0
    timeouts = 0

    total_reward = 0.0
    total_steps = 0
    max_rows = []

    action_counts = Counter()

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(
            seed=EVAL_SEED_START + episode
        )

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

        elif info.get("collision", False):
            collisions += 1

        elif truncated:
            timeouts += 1

    total_actions = sum(action_counts.values())

    result = {
        "name": name,
        "success": successes / NUM_EPISODES,
        "reward": total_reward / NUM_EPISODES,
        "max_row": sum(max_rows) / NUM_EPISODES,
        "episode_length": total_steps / NUM_EPISODES,
        "collisions": collisions,
        "timeouts": timeouts,
    }

    for action, action_name in ACTION_NAMES.items():
        result[f"{action_name}_pct"] = (
            action_counts[action] / total_actions
            if total_actions
            else 0.0
        )

    env.close()

    return result


results = []

for name, model_path in CONFIGS:
    if not Path(model_path + ".zip").exists():
        print(f"WARNING: missing {model_path}.zip")
        continue

    result = evaluate(name, model_path)
    results.append(result)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Success rate: {result['success']:.1%}")
    print(f"Average reward: {result['reward']:.3f}")
    print(f"Average max row: {result['max_row']:.2f}")
    print(f"Average episode length: {result['episode_length']:.2f}")
    print(f"Collisions: {result['collisions']}")
    print(f"Timeouts: {result['timeouts']}")

    print()
    print("Action distribution:")

    for action_name in ACTION_NAMES.values():
        print(
            f"  {action_name:>8}: "
            f"{result[f'{action_name}_pct']:.2%}"
        )


print()
print("=" * 92)
print("A2C n_steps SWEEP SUMMARY")
print("=" * 92)

print(
    f"{'Configuration':<18}"
    f"{'Success':>10}"
    f"{'Reward':>10}"
    f"{'Max row':>10}"
    f"{'Wait':>9}"
    f"{'Forward':>10}"
    f"{'Backward':>10}"
    f"{'Left':>9}"
    f"{'Right':>9}"
)

for r in results:
    print(
        f"{r['name']:<18}"
        f"{r['success']:>9.1%}"
        f"{r['reward']:>10.3f}"
        f"{r['max_row']:>10.2f}"
        f"{r['wait_pct']:>8.1%}"
        f"{r['forward_pct']:>9.1%}"
        f"{r['backward_pct']:>9.1%}"
        f"{r['left_pct']:>8.1%}"
        f"{r['right_pct']:>8.1%}"
    )
