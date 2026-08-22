import numpy as np

from crossyroad_rl.env_v8 import CrossyRoadEnvV8


# ============================================================
# TEST 1:
# Unsupported river position must drown.
# ============================================================

env = CrossyRoadEnvV8()
env.reset(seed=0)

env.player_y = 5

# Find a point that is definitely unsupported.
unsupported_x = None

for x in np.linspace(
    0.0,
    env.grid_width - 0.01,
    500,
):
    env.player_x = float(x)

    if env._get_supporting_platform() is None:
        unsupported_x = float(x)
        break

assert unsupported_x is not None

env.player_x = unsupported_x

(
    obs,
    reward,
    terminated,
    truncated,
    info,
) = env.step(0)

assert terminated
assert not truncated
assert info["drowned"]
assert info["collision"]
assert info["failure_type"] == "drowned"

print(
    "PASS: unsupported river position drowns."
)


# ============================================================
# TEST 2:
# Supported river position survives.
# ============================================================

env = CrossyRoadEnvV8()
env.reset(seed=1)

env.player_y = 5

platform = env.rivers[5][0]

env.player_x = platform["x"]

(
    obs,
    reward,
    terminated,
    truncated,
    info,
) = env.step(0)

assert not info["drowned"]

print(
    "PASS: supported river position survives."
)


# ============================================================
# TEST 3:
# Platform carries player.
# ============================================================

env = CrossyRoadEnvV8()
env.reset(seed=2)

env.player_y = 5

platform = env.rivers[5][0]

# Avoid a wrap-edge case for this unit test by selecting
# any platform comfortably inside the map.
for candidate in env.rivers[5]:
    projected = (
        candidate["x"]
        + candidate["speed"]
        * env.physics_dt
        * env.physics_ticks_per_action
    )

    if (
        0.5
        < projected
        < env.grid_width - 0.5
    ):
        platform = candidate
        break

env.player_x = platform["x"]

start_x = env.player_x
speed = platform["speed"]

(
    obs,
    reward,
    terminated,
    truncated,
    info,
) = env.step(0)

expected_x = (
    start_x
    + speed
    * env.physics_dt
    * env.physics_ticks_per_action
)

assert not info["drowned"]

assert np.isclose(
    env.player_x,
    expected_x,
)

print(
    "PASS: platform carries player."
)

print()
print(
    "All V8 river mechanics checks passed."
)
