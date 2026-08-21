import argparse
import csv
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import QRDQN, TRPO

from crossyroad_rl.env import CrossyRoadEnv
from crossyroad_rl.env_v4 import CrossyRoadEnvV4
from crossyroad_rl.env_v5 import CrossyRoadEnvV5


ENVIRONMENTS = {
    "v3": CrossyRoadEnv,
    "v4": CrossyRoadEnvV4,
    "v5": CrossyRoadEnvV5,
}


ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
    "qrdqn": QRDQN,
    "a2c": A2C,
    "trpo": TRPO,
}

ACTION_NAMES = {
    0: "wait",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
}


def read_run_metadata(run_dir):
    metadata_path = run_dir / "run_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing run metadata: {metadata_path}"
        )

    with metadata_path.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)

    return {
        "algorithm": row["algorithm"],
        "seed": int(row["seed"]),
        "timesteps": int(row["timesteps"]),
        "checkpoint_freq": int(row["checkpoint_freq"]),
        "wall_time_seconds": float(row["wall_time_seconds"]),
    }


def evaluate_model(
    env_name,
    algorithm,
    seed,
    checkpoint_steps,
    model_path,
    num_episodes,
    eval_seed_start,
):
    model_class = ALGORITHMS[algorithm]

    env_class = ENVIRONMENTS[env_name]
    env = env_class()

    model = model_class.load(str(model_path))

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

    for episode in range(num_episodes):
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

            obs, reward, terminated, truncated, info = env.step(
                action
            )

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
                collision_rows[env.player_y] += 1

            elif truncated:
                timeouts += 1

    total_actions = sum(action_counts.values())

    def action_pct(action):
        if total_actions == 0:
            return 0.0

        return action_counts[action] / total_actions

    avg_success_length = (
        sum(successful_steps) / len(successful_steps)
        if successful_steps
        else 0.0
    )

    avg_failure_length = (
        sum(failed_steps) / len(failed_steps)
        if failed_steps
        else 0.0
    )

    return {
        "environment": env_name,
        "algorithm": algorithm,
        "training_seed": seed,
        "checkpoint_steps": checkpoint_steps,
        "evaluation_episodes": num_episodes,
        "eval_seed_start": eval_seed_start,

        "successes": successes,
        "success_rate": successes / num_episodes,

        "collisions": collisions,
        "collision_rate": collisions / num_episodes,

        "timeouts": timeouts,
        "timeout_rate": timeouts / num_episodes,

        "avg_reward": total_reward / num_episodes,
        "avg_episode_length": total_steps / num_episodes,
        "avg_max_row": sum(max_rows) / num_episodes,

        "avg_success_episode_length": avg_success_length,
        "avg_failure_episode_length": avg_failure_length,

        "wait_pct": action_pct(0),
        "forward_pct": action_pct(1),
        "backward_pct": action_pct(2),
        "left_pct": action_pct(3),
        "right_pct": action_pct(4),

        "collision_row_1": collision_rows.get(1, 0),
        "collision_row_3": collision_rows.get(3, 0),

        "model_path": str(model_path),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["v3", "v4", "v5"],
        default="v3",
        help="Environment version used for evaluation.",
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["ppo", "dqn", "qrdqn", "a2c", "trpo"],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
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

    if args.algorithm == "qrdqn":
        run_name = f"qrdqn_50q_seed{args.seed}"
    else:
        run_name = f"{args.algorithm}_seed{args.seed}"
    if args.env == "v3":
        # Preserve compatibility with existing v3 runs.
        run_dir = Path("results/runs") / run_name
    else:
        run_dir = Path("results/runs") / args.env / run_name

    metadata = read_run_metadata(run_dir)

    total_timesteps = metadata["timesteps"]
    checkpoint_freq = metadata["checkpoint_freq"]

    checkpoints = list(
        range(
            checkpoint_freq,
            total_timesteps + 1,
            checkpoint_freq,
        )
    )

    # If the final training step is not exactly divisible by
    # checkpoint_freq, also evaluate final_model at its true step count.
    if total_timesteps not in checkpoints:
        checkpoints.append(total_timesteps)

    rows = []

    for steps in checkpoints:

        # The final model always corresponds to total_timesteps.
        if steps == total_timesteps:
            model_path = run_dir / "final_model"

        else:
            model_path = (
                run_dir
                / "checkpoints"
                / f"{run_name}_{steps}_steps"
            )

        zip_path = Path(str(model_path) + ".zip")

        if not zip_path.exists():
            print(
                f"Skipping {steps}: "
                f"{zip_path} does not exist."
            )
            continue

        print(
            f"Evaluating "
            f"{args.algorithm} "
            f"seed={args.seed} "
            f"steps={steps}..."
        )

        result = evaluate_model(
            env_name=args.env,
            algorithm=args.algorithm,
            seed=args.seed,
            checkpoint_steps=steps,
            model_path=model_path,
            num_episodes=args.episodes,
            eval_seed_start=args.eval_seed_start,
        )

        rows.append(result)

        print(
            f"  success={result['success_rate']:.1%} "
            f"reward={result['avg_reward']:.3f} "
            f"max_row={result['avg_max_row']:.2f} "
            f"forward={result['forward_pct']:.1%}"
        )

    if not rows:
        raise SystemExit("No checkpoints found.")

    output_path = run_dir / "evaluation.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
