from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from crossyroad_rl.env import CrossyRoadEnv


NUM_EPISODES = 100
EVAL_SEED_START = 1000

MODELS = [
    (
        "200k",
        "results/runs/recurrent_ppo_seed0/checkpoints/"
        "recurrent_ppo_seed0_200000_steps",
    ),
    (
        "400k",
        "results/runs/recurrent_ppo_seed0/checkpoints/"
        "recurrent_ppo_seed0_400000_steps",
    ),
    (
        "600k",
        "results/runs/recurrent_ppo_seed0/checkpoints/"
        "recurrent_ppo_seed0_600000_steps",
    ),
    (
        "800k",
        "results/runs/recurrent_ppo_seed0/checkpoints/"
        "recurrent_ppo_seed0_800000_steps",
    ),
    (
        "1m",
        "results/runs/recurrent_ppo_seed0/final_model",
    ),
]

ACTION_NAMES = {
    0: "wait",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
}


def evaluate(checkpoint, model_path):
    env = CrossyRoadEnv()
    model = RecurrentPPO.load(model_path)

    successes = 0
    collisions = 0
    timeouts = 0

    total_reward = 0.0
    total_steps = 0
    max_rows = []

    action_counts = Counter()
    collision_rows = Counter()

    successful_steps = []
    failed_steps = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(
            seed=EVAL_SEED_START + episode
        )

        # Reset recurrent state at the beginning of every episode.
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        episode_reward = 0.0
        episode_steps = 0

        while True:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )

            action = int(action)
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            done = terminated or truncated

            # For the next prediction, tell RecurrentPPO whether
            # the previous transition ended an episode.
            episode_start = np.array(
                [done],
                dtype=bool,
            )

            if done:
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
                collision_rows[env.player_y] += 1

            elif truncated:
                timeouts += 1

    total_actions = sum(action_counts.values())

    def action_pct(action):
        if total_actions == 0:
            return 0.0
        return action_counts[action] / total_actions

    print()
    print("=" * 72)
    print(f"Recurrent PPO checkpoint: {checkpoint}")
    print("=" * 72)

    print(f"Episodes: {NUM_EPISODES}")
    print(f"Successes: {successes}")
    print(f"Collisions: {collisions}")
    print(f"Timeouts: {timeouts}")
    print(f"Success rate: {successes / NUM_EPISODES:.2%}")
    print(f"Average reward: {total_reward / NUM_EPISODES:.3f}")
    print(f"Average episode length: {total_steps / NUM_EPISODES:.2f}")
    print(f"Average max row: {sum(max_rows) / NUM_EPISODES:.2f}")

    print()
    print("Action distribution:")

    for action in range(5):
        print(
            f"  {ACTION_NAMES[action]:>8}: "
            f"{action_counts[action]:5d} "
            f"({action_pct(action):6.2%})"
        )

    print()
    print("Collision rows:")

    if collision_rows:
        for row in sorted(collision_rows):
            print(
                f"  row {row}: "
                f"{collision_rows[row]} collisions"
            )
    else:
        print("  None")

    if successful_steps:
        print(
            "\nAverage successful episode length: "
            f"{sum(successful_steps) / len(successful_steps):.2f}"
        )

    if failed_steps:
        print(
            "Average failed episode length: "
            f"{sum(failed_steps) / len(failed_steps):.2f}"
        )

    return {
        "checkpoint": checkpoint,
        "success_rate": successes / NUM_EPISODES,
        "avg_reward": total_reward / NUM_EPISODES,
        "avg_length": total_steps / NUM_EPISODES,
        "avg_max_row": sum(max_rows) / NUM_EPISODES,
        "collision_rate": collisions / NUM_EPISODES,
        "timeout_rate": timeouts / NUM_EPISODES,
        "wait_pct": action_pct(0),
        "forward_pct": action_pct(1),
        "backward_pct": action_pct(2),
        "left_pct": action_pct(3),
        "right_pct": action_pct(4),
    }


results = []

for checkpoint, model_path in MODELS:
    if not Path(model_path + ".zip").exists():
        print(f"WARNING: missing {model_path}.zip")
        continue

    results.append(
        evaluate(
            checkpoint,
            model_path,
        )
    )


print()
print("=" * 92)
print("RECURRENT PPO V3 LEARNING CURVE SUMMARY")
print("=" * 92)

print(
    f"{'Steps':<10}"
    f"{'Success':>10}"
    f"{'Reward':>10}"
    f"{'Length':>10}"
    f"{'Max row':>10}"
    f"{'Forward':>10}"
    f"{'Timeouts':>10}"
)

for r in results:
    print(
        f"{r['checkpoint']:<10}"
        f"{r['success_rate']:>9.1%}"
        f"{r['avg_reward']:>10.3f}"
        f"{r['avg_length']:>10.2f}"
        f"{r['avg_max_row']:>10.2f}"
        f"{r['forward_pct']:>9.1%}"
        f"{r['timeout_rate'] * NUM_EPISODES:>10.0f}"
    )
