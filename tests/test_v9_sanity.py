import numpy as np

from crossyroad_rl.env_v9 import CrossyRoadEnvV9


def run_policy(policy_fn, episodes=100, seed_start=1000):
    successes = 0
    road_collisions = 0
    drownings = 0
    timeouts = 0

    max_rows = []
    lengths = []

    for episode in range(episodes):
        env = CrossyRoadEnvV9()
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

        elif info.get("road_collision", False):
            road_collisions += 1

        elif info.get("drowned", False):
            drownings += 1

        else:
            timeouts += 1

        env.close()

    return {
        "successes": successes,
        "success_rate": successes / episodes,
        "road_collisions": road_collisions,
        "drownings": drownings,
        "timeouts": timeouts,
        "avg_max_row": np.mean(max_rows),
        "avg_length": np.mean(lengths),
    }


def random_policy(env):
    return env.action_space.sample()


def heuristic_policy(env):
    """
    One-step mixed-hazard heuristic.

    - Safe row ahead: move forward.
    - Road ahead: predict car positions over one full RL action
      and cross only when current x is reasonably safe.
    - River ahead: predict platform positions over one full action.
      Cross if current x is supported; otherwise move horizontally
      toward the nearest predicted platform.
    """

    next_row = min(
        env.player_y + 1,
        env.goal_y,
    )

    hazard_type = env.hazard_types.get(next_row)

    if hazard_type is None:
        return 1

    total_dt = (
        env.physics_ticks_per_action
        * env.physics_dt
    )

    # --------------------------------------------------------
    # ROAD
    # --------------------------------------------------------

    if hazard_type == "road":
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
                return 0

        return 1

    # --------------------------------------------------------
    # RIVER
    # --------------------------------------------------------

    platforms = env.rivers[next_row]
    half_width = env.river_half_widths[next_row]

    predicted = []

    for platform in platforms:
        x = (
            platform["x"]
            + platform["speed"] * total_dt
        ) % env.grid_width

        predicted.append(x)

    # Cross if current x should land on a platform.
    for x in predicted:
        if (
            env._circular_distance(
                env.player_x,
                x,
            )
            <= half_width
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

    if abs(direct) <= env.grid_width / 2:
        return 4 if direct > 0 else 3

    return 3 if direct > 0 else 4


print("V9 RANDOM POLICY")
print("=" * 60)

result = run_policy(
    random_policy,
    episodes=100,
)

for key, value in result.items():
    print(f"{key}: {value}")


print()
print("V9 MIXED-HAZARD HEURISTIC")
print("=" * 60)

result = run_policy(
    heuristic_policy,
    episodes=100,
)

for key, value in result.items():
    print(f"{key}: {value}")
