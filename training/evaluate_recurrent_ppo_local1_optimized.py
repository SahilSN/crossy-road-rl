from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


ACTION_NAMES = [
    "wait",
    "forward",
    "backward",
    "left",
    "right",
]


def evaluate(
    model_path,
    episodes,
    seed_start,
):
    model = RecurrentPPO.load(
        str(model_path)
    )

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

    for episode in range(episodes):
        obs, _ = env.reset(
            seed=seed_start + episode
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
                collision_rows.get(row, 0)
                + 1
            )

        else:
            timeouts += 1

    env.close()

    total_actions = action_counts.sum()

    result = {
        "evaluation_episodes": episodes,
        "eval_seed_start": seed_start,

        "successes": successes,
        "success_rate": successes / episodes,

        "collisions": collisions,
        "collision_rate": collisions / episodes,

        "timeouts": timeouts,
        "timeout_rate": timeouts / episodes,

        "avg_reward": np.mean(rewards),
        "avg_episode_length": np.mean(lengths),
        "avg_max_row": np.mean(max_rows),
    }

    for i, name in enumerate(ACTION_NAMES):
        result[f"{name}_pct"] = (
            action_counts[i] / total_actions
            if total_actions
            else 0.0
        )

    return result, collision_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    run_name = (
        f"recurrent_ppo_64_seed{args.seed}"
    )

    run_dir = (
        Path("results/runs/v6_local1")
        / run_name
    )

    checkpoints = {
        200_000: (
            run_dir
            / "checkpoints"
            / f"{run_name}_200000_steps.zip"
        ),
        400_000: (
            run_dir
            / "checkpoints"
            / f"{run_name}_400000_steps.zip"
        ),
        600_000: (
            run_dir
            / "checkpoints"
            / f"{run_name}_600000_steps.zip"
        ),
        800_000: (
            run_dir
            / "checkpoints"
            / f"{run_name}_800000_steps.zip"
        ),
        1_000_000: (
            run_dir
            / "checkpoints"
            / f"{run_name}_1000000_steps.zip"
        ),
    }

    rows = []

    for steps, path in checkpoints.items():
        if not path.exists():
            print(
                f"Missing checkpoint: {path}"
            )
            continue

        result, collision_rows = evaluate(
            path,
            args.episodes,
            args.eval_seed_start,
        )

        row = {
            "environment": "v6_local1",
            "algorithm": "recurrent_ppo",
            "config": "lstm64_ent02",
            "training_seed": args.seed,
            "checkpoint_steps": steps,
            **result,
        }

        rows.append(row)

        print()
        print("=" * 72)
        print(
            f"Recurrent PPO 64 | "
            f"seed {args.seed} | "
            f"{steps // 1000}k"
        )
        print("=" * 72)

        print(
            f"Success:   "
            f"{result['success_rate']:.1%}"
        )

        print(
            f"Reward:    "
            f"{result['avg_reward']:.3f}"
        )

        print(
            f"Max row:   "
            f"{result['avg_max_row']:.2f}"
        )

        print(
            f"Collision: "
            f"{result['collision_rate']:.1%}"
        )

        print(
            f"Timeout:   "
            f"{result['timeout_rate']:.1%}"
        )

        print("\nActions:")

        for name in ACTION_NAMES:
            print(
                f"  {name:>8}: "
                f"{result[f'{name}_pct']:.1%}"
            )

        print(
            "\nCollision rows:",
            collision_rows
            if collision_rows
            else "None",
        )

    if not rows:
        raise SystemExit(
            "No checkpoints were evaluated."
        )

    df = pd.DataFrame(rows)

    output_path = (
        run_dir / "evaluation.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
