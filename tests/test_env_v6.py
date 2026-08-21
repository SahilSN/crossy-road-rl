from stable_baselines3.common.env_checker import check_env

from crossyroad_rl.env_v6 import CrossyRoadEnvV6


for horizon in [1, 2, 3]:
    env = CrossyRoadEnvV6(
        observation_horizon=horizon
    )

    print()
    print(f"local{horizon}")
    print(
        "Observation shape:",
        env.observation_space.shape,
    )
    print(
        "Action space:",
        env.action_space,
    )

    check_env(
        env,
        warn=True,
    )

    print("Environment check passed.")
