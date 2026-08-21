import numpy as np

from crossyroad_rl.env_v5 import CrossyRoadEnvV5
from crossyroad_rl.env_v6 import CrossyRoadEnvV6


ACTIONS = [
    0, 1, 0, 1, 3,
    0, 4, 1, 0, 1,
    2, 0, 1, 1, 0,
]


for seed in range(20):
    env5 = CrossyRoadEnvV5()

    env6 = CrossyRoadEnvV6(
        observation_horizon=2
    )

    obs5, info5 = env5.reset(seed=seed)
    obs6, info6 = env6.reset(seed=seed)

    assert np.array_equal(
        obs5,
        obs6,
    )

    assert info5 == info6

    for action in ACTIONS:
        result5 = env5.step(action)
        result6 = env6.step(action)

        (
            obs5,
            reward5,
            terminated5,
            truncated5,
            info5,
        ) = result5

        (
            obs6,
            reward6,
            terminated6,
            truncated6,
            info6,
        ) = result6

        assert np.array_equal(
            obs5,
            obs6,
        )

        assert reward5 == reward6
        assert terminated5 == terminated6
        assert truncated5 == truncated6
        assert info5 == info6

        if terminated5 or truncated5:
            break

    print(
        f"seed {seed}: "
        "v5 == v6_local2"
    )


print()
print(
    "v6_local2 exactly reproduces v5."
)
