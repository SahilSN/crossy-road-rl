from collections import Counter

from crossyroad_rl.env_v4 import CrossyRoadEnvV4


NUM_EPISODES = 100


def circular_distance(x1, x2, width):
    direct = abs(x1 - x2)
    wrapped = width - direct
    return min(direct, wrapped)


def road_looks_safe(env, row):
    """
    Conservative one-step safety check.

    Estimate where each car will be after one full RL action
    (all internal physics ticks), and only cross if no car is
    too close to the player's x-position.
    """

    if row not in env.road_rows:
        return True

    total_dt = (
        env.physics_ticks_per_action
        * env.physics_dt
    )

    for car in env.lanes[row]:
        future_x = (
            car["x"]
            + car["speed"] * total_dt
        ) % env.grid_width

        distance = circular_distance(
            env.player_x,
            future_x,
            env.grid_width,
        )

        # Slightly more conservative than the actual
        # collision threshold.
        if distance < 1.0:
            return False

    return True


env = CrossyRoadEnvV4()

successes = 0
collisions = 0
timeouts = 0

max_rows = []
episode_lengths = []
collision_rows = Counter()


for episode in range(NUM_EPISODES):
    obs, info = env.reset(seed=1000 + episode)

    steps = 0

    while True:
        next_row = env.player_y + 1

        if next_row <= env.goal_y and road_looks_safe(
            env,
            next_row,
        ):
            action = 1  # forward
        else:
            action = 0  # wait

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        steps += 1

        if terminated or truncated:
            break

    max_rows.append(info["max_y"])
    episode_lengths.append(steps)

    if info["success"]:
        successes += 1

    elif info["collision"]:
        collisions += 1
        collision_rows[env.player_y] += 1

    elif truncated:
        timeouts += 1


print()
print("=== Heuristic-policy v4 sanity check ===")
print(f"Episodes: {NUM_EPISODES}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts: {timeouts}")

print(
    "Success rate:",
    f"{successes / NUM_EPISODES:.1%}",
)

print(
    "Average max row:",
    sum(max_rows) / NUM_EPISODES,
)

print(
    "Average episode length:",
    sum(episode_lengths) / NUM_EPISODES,
)

print()
print("Collision rows:")

if collision_rows:
    for row in sorted(collision_rows):
        print(
            f"  row {row}: "
            f"{collision_rows[row]}"
        )
else:
    print("  None")
