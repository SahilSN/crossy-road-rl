from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


RUN_DIR = Path(
    "results/runs/v6_local1_recurrent_tuning/"
    "recurrent_ppo_ent02_seed0"
)

EPISODES = 100
EVAL_SEED_START = 1000

ACTION_NAMES = [
    "wait",
    "forward",
    "backward",
    "left",
    "right",
]


def evaluate(model_path):
    model = RecurrentPPO.load(str(model_path))

    env = CrossyRoadEnvV6(
        observation_horizon=1
    )

    successes = 0
    collisions = 0
    timeouts = 0

    rewards = []
    lengths = []
    max_rows = []

    action_counts = np.zeros(
        len(ACTION_NAMES),
        dtype=int,
    )

    collision_rows = {}

    for episode in range(EPISODES):
        obs, _ = env.reset(
            seed=EVAL_SEED_START + episode
        )

        lstm_state = None
        episode_start = np.ones(
            (1,),
            dtype=bool,
        )

        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )

            action = int(action)

            action_counts[action] += 1

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward
            length += 1

            done = terminated or truncated

            episode_start = np.array(
                [done],
                dtype=bool,
            )

        rewards.append(total_reward)
        lengths.append(length)
        max_rows.append(env.max_y)

        if info.get("success", False):
            successes += 1

        elif info.get("collision", False):
            collisions += 1

            row = env.player_y

            collision_rows[row] = (
                collision_rows.get(row, 0) + 1
            )

        else:
            timeouts += 1

    env.close()

    total_actions = action_counts.sum()

    result = {
        "success_rate": successes / EPISODES,
        "avg_reward": np.mean(rewards),
        "avg_episode_length": np.mean(lengths),
        "avg_max_row": np.mean(max_rows),
        "collision_rate": collisions / EPISODES,
        "timeout_rate": timeouts / EPISODES,
    }

    for i, action_name in enumerate(ACTION_NAMES):
        result[f"{action_name}_pct"] = (
            action_counts[i] / total_actions
        )

    return result, collision_rows


candidates = {
    200_000: (
        RUN_DIR
        / "checkpoints"
        / "recurrent_ppo_ent02_seed0_200000_steps.zip"
    ),
    400_000: (
        RUN_DIR
        / "checkpoints"
        / "recurrent_ppo_ent02_seed0_400000_steps.zip"
    ),
    600_000: (
        RUN_DIR
        / "checkpoints"
        / "recurrent_ppo_ent02_continued_600000_steps.zip"
    ),
    800_000: (
        RUN_DIR
        / "checkpoints"
        / "recurrent_ppo_ent02_continued_800000_steps.zip"
    ),
    1_000_000: (
        RUN_DIR
        / "checkpoints"
        / "recurrent_ppo_ent02_continued_1000000_steps.zip"
    ),
}

rows = []

for steps, path in candidates.items():
    if not path.exists():
        print(f"Missing checkpoint: {path}")
        continue

    result, collision_rows = evaluate(path)

    rows.append({
        "steps": steps,
        **result,
    })

    print()
    print("=" * 72)
    print(f"Recurrent PPO ent02 | {steps // 1000}k")
    print("=" * 72)

    print(f"Success:   {result['success_rate']:.1%}")
    print(f"Reward:    {result['avg_reward']:.3f}")
    print(f"Max row:   {result['avg_max_row']:.2f}")
    print(f"Collision: {result['collision_rate']:.1%}")
    print(f"Timeout:   {result['timeout_rate']:.1%}")

    print("\nActions:")

    for action in ACTION_NAMES:
        print(
            f"  {action:>8}: "
            f"{result[f'{action}_pct']:.1%}"
        )

    print(
        "\nCollision rows:",
        collision_rows if collision_rows else "None",
    )


df = pd.DataFrame(rows)

print()
print("=" * 90)
print("RECURRENT PPO ENT02 LOCAL1 LEARNING CURVE")
print("=" * 90)

print(
    df.to_string(
        index=False,
        formatters={
            "success_rate": lambda x: f"{x:.1%}",
            "avg_reward": lambda x: f"{x:.3f}",
            "avg_episode_length": lambda x: f"{x:.2f}",
            "avg_max_row": lambda x: f"{x:.2f}",
            "collision_rate": lambda x: f"{x:.1%}",
            "timeout_rate": lambda x: f"{x:.1%}",
            "wait_pct": lambda x: f"{x:.1%}",
            "forward_pct": lambda x: f"{x:.1%}",
            "backward_pct": lambda x: f"{x:.1%}",
            "left_pct": lambda x: f"{x:.1%}",
            "right_pct": lambda x: f"{x:.1%}",
        },
    )
)

df.to_csv(
    RUN_DIR / "evaluation_1m.csv",
    index=False,
)
