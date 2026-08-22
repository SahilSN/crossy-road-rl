from crossyroad_rl.env_v7 import CrossyRoadEnvV7


def layout_for_seed(seed):
    env = CrossyRoadEnvV7()
    env.reset(seed=seed)

    rows = tuple(env.road_rows)

    env.close()
    return rows


print("V7 PROCEDURAL LAYOUTS")
print("=" * 60)

layouts = {}

for seed in range(20):
    layout = layout_for_seed(seed)
    layouts[seed] = layout

    print(f"seed {seed:2d}: {layout}")


print()
print("Unique layouts:", len(set(layouts.values())))


# Determinism
for seed in range(20):
    first = layout_for_seed(seed)
    second = layout_for_seed(seed)

    assert first == second, (
        f"Seed {seed} is not deterministic: "
        f"{first} != {second}"
    )

print("Deterministic seed test passed.")


# Layout constraints
for seed, rows in layouts.items():
    assert len(rows) == 4
    assert len(set(rows)) == 4

    assert all(
        1 <= row <= 8
        for row in rows
    )

    assert any(
        row <= 4
        for row in rows
    )

    assert any(
        row >= 5
        for row in rows
    )

    max_run = 1
    current_run = 1

    for i in range(1, len(rows)):
        if rows[i] == rows[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    assert max_run <= 2, (
        f"Seed {seed} produced >2 consecutive roads: {rows}"
    )

print("Layout constraint checks passed.")
