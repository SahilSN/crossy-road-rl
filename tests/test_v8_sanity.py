import numpy as np

from crossyroad_rl.env_v8 import CrossyRoadEnvV8


def run_policy(policy_fn, episodes=100, seed_start=1000):
    successes = 0
    collisions = 0
    drowned = 0
    road_collisions = 0
    timeouts = 0
    max_rows = []
    lengths = []

    for episode in range(episodes):
        env = CrossyRoadEnvV8()
        obs, info = env.reset(seed=seed_start + episode)

        done = False
        length = 0

        while not done:
            action = policy_fn(env)

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            length += 1
            done = terminated or truncated

        max_rows.append(env.max_y)
        lengths.append(length)

        if info.get("success", False):
            successes += 1

        elif info.get("drowned", False):
            drowned += 1
            collisions += 1

        elif info.get("road_collision", False):
            road_collisions += 1
            collisions += 1

        else:
            timeouts += 1

        env.close()

    return {
        "successes": successes,
        "success_rate": successes / episodes,
        "collisions": collisions,
        "road_collisions": road_collisions,
        "drowned": drowned,
        "timeouts": timeouts,
        "avg_max_row": np.mean(max_rows),
        "avg_length": np.mean(lengths),
    }


def random_policy(env):
    return env.action_space.sample()


def heuristic_policy(env):
    """
    Simple one-step local heuristic.

    Safe rows:
        move forward.

    Road ahead:
        move forward only if the target x looks safe after
        the next full action's worth of car movement;
        otherwise wait.

    River ahead:
        move forward only if a platform is predicted to
        support the current x after the next full action;
        otherwise move horizontally toward the nearest
        predicted platform.
    """

    next_row = min(
        env.player_y + 1,
        env.goal_y,
    )

    # Goal/safe row.
    if (
        next_row not in env.road_rows
        and next_row not in env.river_rows
    ):
        return 1

    total_dt = (
        env.physics_ticks_per_action
        * env.physics_dt
    )

    # --------------------------------------------------------
    # Road
    # --------------------------------------------------------

    if next_row in env.road_rows:
        safe = True

        for car in env.lanes[next_row]:
            predicted_x = (
                car["x"]
                + car["speed"] * total_dt
            ) % env.grid_width

            distance = env._circular_distance(
                env.player_x,
                predicted_x,
            )

            if distance < 0.8:
                safe = False
                break

        return 1 if safe else 0

    # --------------------------------------------------------
    # River
    # --------------------------------------------------------

    platforms = env.rivers[next_row]

    predicted = []

    for platform in platforms:
        x = (
            platform["x"]
            + platform["speed"] * total_dt
        ) % env.grid_width

        predicted.append(x)

    # Can we safely move forward at current x?
    for x in predicted:
        if (
            env._circular_distance(
                env.player_x,
                x,
            )
            <= env.platform_half_width
        ):
            return 1

    # Otherwise move toward nearest predicted platform.
    target = min(
        predicted,
        key=lambda x: env._circular_distance(
            env.player_x,
            x,
        ),
    )

    direct = target - env.player_x

    # Handle wrap-aware shortest direction.
    if abs(direct) <= env.grid_width / 2:
        return 4 if direct > 0 else 3

    # Wrapped direction is shorter.
    return 3 if direct > 0 else 4


print("V8 RANDOM POLICY")
print("=" * 60)

random_result = run_policy(
    random_policy,
    episodes=100,
)

for k, v in random_result.items():
    print(f"{k}: {v}")


print()
print("V8 PLATFORM-AWARE HEURISTIC")
print("=" * 60)

heuristic_result = run_policy(
    heuristic_policy,
    episodes=100,
)

for k, v in heuristic_result.items():
    print(f"{k}: {v}")
