import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO, QRDQN

from crossyroad_rl.env_v8 import CrossyRoadEnvV8
from crossyroad_rl.env_v9 import CrossyRoadEnvV9


ALGORITHMS = {
    "ppo": PPO,
    "trpo": TRPO,
    "dqn": DQN,
    "qrdqn": QRDQN,
}

ENVIRONMENTS = {
    "v8": CrossyRoadEnvV8,
    "v9": CrossyRoadEnvV9,
}


def run_name(algorithm, seed):
    if algorithm == "qrdqn":
        return f"qrdqn_50q_seed{seed}"
    return f"{algorithm}_seed{seed}"


def get_run_dir(env_name, algorithm, seed):
    name = run_name(algorithm, seed)

    return (
        Path("results/runs")
        / env_name
        / name
    )


def get_model_path(
    train_env,
    algorithm,
    seed,
    checkpoint_steps,
):
    run_dir = get_run_dir(
        train_env,
        algorithm,
        seed,
    )

    if checkpoint_steps == 1_000_000:
        return run_dir / "final_model"

    return (
        run_dir
        / "checkpoints"
        / (
            f"{run_name(algorithm, seed)}"
            f"_{checkpoint_steps}_steps"
        )
    )


def evaluate(
    train_env,
    eval_env,
    algorithm,
    seed,
    checkpoint_steps,
    episodes,
    eval_seed_start,
):
    model_class = ALGORITHMS[algorithm]

    model_path = get_model_path(
        train_env,
        algorithm,
        seed,
        checkpoint_steps,
    )

    zip_path = Path(
        str(model_path) + ".zip"
    )

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Model not found: {zip_path}"
        )

    env = ENVIRONMENTS[eval_env]()

    model = model_class.load(
        str(model_path)
    )

    successes = 0
    collisions = 0
    road_collisions = 0
    drownings = 0
    timeouts = 0

    total_reward = 0.0
    total_steps = 0
    max_rows = []

    action_counts = Counter()

    for episode in range(episodes):
        obs, info = env.reset(
            seed=eval_seed_start + episode
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

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

        total_reward += episode_reward
        total_steps += episode_steps
        max_rows.append(env.max_y)

        if info.get("success", False):
            successes += 1

        elif info.get("collision", False):
            collisions += 1

            if info.get(
                "road_collision",
                False,
            ):
                road_collisions += 1

            if info.get(
                "drowned",
                False,
            ):
                drownings += 1

        elif truncated:
            timeouts += 1

    env.close()

    total_actions = sum(
        action_counts.values()
    )

    def action_pct(action):
        if total_actions == 0:
            return 0.0

        return (
            action_counts[action]
            / total_actions
        )

    return {
        "train_env": train_env,
        "eval_env": eval_env,
        "algorithm": algorithm,
        "training_seed": seed,
        "checkpoint_steps": checkpoint_steps,

        "episodes": episodes,

        "success_rate":
            successes / episodes,

        "collision_rate":
            collisions / episodes,

        "road_collision_rate":
            road_collisions / episodes,

        "drowning_rate":
            drownings / episodes,

        "timeout_rate":
            timeouts / episodes,

        "avg_reward":
            total_reward / episodes,

        "avg_steps":
            total_steps / episodes,

        "avg_max_row":
            sum(max_rows) / len(max_rows),

        "wait_pct":
            action_pct(0),

        "forward_pct":
            action_pct(1),

        "backward_pct":
            action_pct(2),

        "left_pct":
            action_pct(3),

        "right_pct":
            action_pct(4),

        "model_path":
            str(model_path),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-env",
        required=True,
        choices=["v8", "v9"],
    )

    parser.add_argument(
        "--eval-env",
        required=True,
        choices=["v8", "v9"],
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[
            "ppo",
            "trpo",
            "dqn",
            "qrdqn",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=1_000_000,
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

    result = evaluate(
        train_env=args.train_env,
        eval_env=args.eval_env,
        algorithm=args.algorithm,
        seed=args.seed,
        checkpoint_steps=args.checkpoint_steps,
        episodes=args.episodes,
        eval_seed_start=args.eval_seed_start,
    )

    output_dir = Path(
        "results/cross_env_transfer"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_dir
        / (
            f"train_{args.train_env}"
            f"_eval_{args.eval_env}"
            f"_{args.algorithm}"
            f"_seed{args.seed}.csv"
        )
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"TRAIN {args.train_env.upper()} "
        f"→ EVAL {args.eval_env.upper()}"
    )

    print(
        f"algorithm={args.algorithm} "
        f"seed={args.seed}"
    )

    print(
        f"success="
        f"{result['success_rate']:.1%}"
    )

    print(
        f"reward="
        f"{result['avg_reward']:.3f}"
    )

    print(
        f"max_row="
        f"{result['avg_max_row']:.2f}"
    )

    print(
        f"road_collision="
        f"{result['road_collision_rate']:.1%}"
    )

    print(
        f"drowning="
        f"{result['drowning_rate']:.1%}"
    )

    print(
        f"timeout="
        f"{result['timeout_rate']:.1%}"
    )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
