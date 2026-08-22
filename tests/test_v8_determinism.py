import numpy as np

from crossyroad_rl.env_v8 import CrossyRoadEnvV8


ACTIONS = [
    0, 1, 0, 1, 3,
    0, 4, 1, 0, 1,
    2, 0, 1, 1, 0,
]


for seed in range(20):
    a = CrossyRoadEnvV8()
    b = CrossyRoadEnvV8()

    obs_a, info_a = a.reset(seed=seed)
    obs_b, info_b = b.reset(seed=seed)

    assert np.array_equal(
        obs_a,
        obs_b,
    )

    assert info_a == info_b

    for action in ACTIONS:
        result_a = a.step(action)
        result_b = b.step(action)

        (
            obs_a,
            reward_a,
            term_a,
            trunc_a,
            info_a,
        ) = result_a

        (
            obs_b,
            reward_b,
            term_b,
            trunc_b,
            info_b,
        ) = result_b

        assert np.array_equal(
            obs_a,
            obs_b,
        )

        assert reward_a == reward_b
        assert term_a == term_b
        assert trunc_a == trunc_b
        assert info_a == info_b

        if term_a or trunc_a:
            break

    a.close()
    b.close()

    print(
        f"seed {seed}: deterministic"
    )


print()
print(
    "V8 trajectory determinism passed."
)
