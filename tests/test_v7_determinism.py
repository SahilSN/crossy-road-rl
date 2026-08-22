import numpy as np

from crossyroad_rl.env_v7 import CrossyRoadEnvV7


ACTIONS = [
    0, 1, 0, 1, 3,
    4, 1, 0, 1, 2,
    1, 0, 4, 1, 0,
]


for seed in range(20):
    env_a = CrossyRoadEnvV7()
    env_b = CrossyRoadEnvV7()

    obs_a, info_a = env_a.reset(seed=seed)
    obs_b, info_b = env_b.reset(seed=seed)

    assert env_a.road_rows == env_b.road_rows
    assert np.array_equal(obs_a, obs_b)
    assert info_a == info_b

    for action in ACTIONS:
        (
            obs_a,
            reward_a,
            term_a,
            trunc_a,
            info_a,
        ) = env_a.step(action)

        (
            obs_b,
            reward_b,
            term_b,
            trunc_b,
            info_b,
        ) = env_b.step(action)

        assert np.array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        assert term_a == term_b
        assert trunc_a == trunc_b
        assert info_a == info_b

        # Also verify the underlying traffic state.
        assert env_a.road_rows == env_b.road_rows

        for row in env_a.road_rows:
            cars_a = env_a.lanes[row]
            cars_b = env_b.lanes[row]

            assert len(cars_a) == len(cars_b)

            for car_a, car_b in zip(cars_a, cars_b):
                assert np.isclose(
                    car_a["x"],
                    car_b["x"],
                )

                assert np.isclose(
                    car_a["speed"],
                    car_b["speed"],
                )

        if term_a or trunc_a:
            break

    env_a.close()
    env_b.close()

    print(f"seed {seed}: deterministic")


print()
print("V7 trajectory determinism passed.")
