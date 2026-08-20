from collections import Counter
from pathlib import Path

from stable_baselines3 import A2C

from crossyroad_rl.env import CrossyRoadEnv


NUM_EPISODES = 100
EVAL_SEED_START = 1000

CONFIGS = [
    ("baseline", "results/runs/a2c_seed0/checkpoints/a2c_seed0_200000_steps"),
    ("lr3e-4_ent0.01", "results/a2c_tuning/a2c_lr3e4_ent001/final_model"),
    ("lr1e-4_ent0.01", "results/a2c_tuning/a2c_lr1e4_ent001/final_model"),
    ("lr3e-4_ent0.02", "results/a2c_tuning/a2c_lr3e4_ent002/final_model"),
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
    collision_rows = Counter()

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
            collision_rows[env.player_y] += 1

        elif truncated:
            timeouts += 1

    total_actions = sum(action_counts.values())

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Success rate: {successes / NUM_EPISODES:.1%}")
    print(f"Average reward: {total_reward / NUM_EPISODES:.3f}")
    print(f"Average episode length: {total_steps / NUM_EPISODES:.2f}")
    print(f"Average max row: {sum(max_rows) / NUM_EPISODES:.2f}")
    print(f"Collisions: {collisions}")
    print(f"Timeouts: {timeouts}")

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

    if collision_rows:
        for row in sorted(collision_rows):
            print(f"  row {row}: {collision_rows[row]}")
    else:
        print("  None")

    return {
        "name": name,
        "success": successes / NUM_EPISODES,
        "reward": total_reward / NUM_EPISODES,
        "max_row": sum(max_rows) / NUM_EPISODES,
        "forward_pct": (
            action_counts[1] / total_actions
            if total_actions
            else 0.0
        ),
        "timeouts": timeouts,
    }


results = []

for name, model_path in CONFIGS:
    if not Path(model_path + ".zip").exists():
        print(f"WARNING: missing {model_path}.zip")
        continue

    results.append(
        evaluate(name, model_path)
    )


print()
print("=" * 76)
print("A2C TUNING SUMMARY")
print("=" * 76)

print(
    f"{'Configuration':<22}"
    f"{'Success':>10}"
    f"{'Reward':>10}"
    f"{'Max row':>10}"
    f"{'Forward':>10}"
    f"{'Timeouts':>10}"
)

for r in results:
    print(
        f"{r['name']:<22}"
        f"{r['success']:>9.1%}"
        f"{r['reward']:>10.3f}"
        f"{r['max_row']:>10.2f}"
        f"{r['forward_pct']:>9.1%}"
        f"{r['timeouts']:>10d}"
    )
