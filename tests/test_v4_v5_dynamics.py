from crossyroad_rl.env_v4 import CrossyRoadEnvV4
from crossyroad_rl.env_v5 import CrossyRoadEnvV5


ACTIONS = [
    0, 0, 1, 0, 1,
    3, 0, 4, 1, 0,
    1, 2, 0, 1, 1,
]


for seed in range(10):
    env4 = CrossyRoadEnvV4()
    env5 = CrossyRoadEnvV5()

    obs4, info4 = env4.reset(seed=seed)
    obs5, info5 = env5.reset(seed=seed)

    for step_number, action in enumerate(ACTIONS):
        result4 = env4.step(action)
        result5 = env5.step(action)

        obs4, reward4, terminated4, truncated4, info4 = result4
        obs5, reward5, terminated5, truncated5, info5 = result5

        assert env4.player_x == env5.player_x
        assert env4.player_y == env5.player_y
        assert env4.max_y == env5.max_y

        assert reward4 == reward5
        assert terminated4 == terminated5
        assert truncated4 == truncated5

        assert info4 == info5

        for row in env4.road_rows:
            cars4 = env4.lanes[row]
            cars5 = env5.lanes[row]

            assert len(cars4) == len(cars5)

            for car4, car5 in zip(cars4, cars5):
                assert car4["x"] == car5["x"]
                assert car4["speed"] == car5["speed"]

        if terminated4 or truncated4:
            break

    print(f"seed {seed}: dynamics match")

print()
print("v4 and v5 dynamics are identical.")
print("Only observation availability differs.")
