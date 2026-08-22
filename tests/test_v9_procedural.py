from collections import Counter

from crossyroad_rl.env_v9 import CrossyRoadEnvV9


layouts = []
type_counts = Counter()


print("V9 PROCEDURAL MIXED LAYOUTS")
print("=" * 75)


for seed in range(30):
    env = CrossyRoadEnvV9()

    env.reset(seed=seed)

    layout = tuple(
        (
            row,
            env.hazard_types[row],
        )
        for row in env.hazard_rows
    )

    layouts.append(layout)

    num_roads = len(env.road_rows)
    num_rivers = len(env.river_rows)

    type_counts[
        (num_roads, num_rivers)
    ] += 1

    print(
        f"seed {seed:2d}: "
        f"{layout}"
    )

    # Exactly four hazards.
    assert len(env.hazard_rows) == 4

    # Both mechanics present.
    assert num_roads >= 1
    assert num_rivers >= 1

    # Hazard placement constraints.
    assert any(
        row <= 4
        for row in env.hazard_rows
    )

    assert any(
        row >= 5
        for row in env.hazard_rows
    )

    max_run = 1
    current_run = 1

    for i in range(
        1,
        len(env.hazard_rows),
    ):
        if (
            env.hazard_rows[i]
            == env.hazard_rows[i - 1] + 1
        ):
            current_run += 1
            max_run = max(
                max_run,
                current_run,
            )
        else:
            current_run = 1

    assert max_run <= 2

    env.close()


print()
print(
    "Unique complete layouts:",
    len(set(layouts)),
)

print()
print(
    "Road/river composition counts:"
)

for composition, count in sorted(
    type_counts.items()
):
    print(
        f"  roads={composition[0]}, "
        f"rivers={composition[1]}: "
        f"{count}"
    )

print()
print(
    "V9 procedural checks passed."
)
