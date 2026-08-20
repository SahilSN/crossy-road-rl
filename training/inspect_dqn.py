import torch

from stable_baselines3 import DQN
from crossyroad_rl.env import CrossyRoadEnv


ACTION_NAMES = [
    "wait",
    "forward",
    "backward",
    "left",
    "right",
]

model = DQN.load("results/dqn_basic_v2_1m")
env = CrossyRoadEnv()

print("DQN Q-values at the initial state")
print("=" * 75)

for seed in range(1000, 1010):
    obs, _ = env.reset(seed=seed)

    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    with torch.no_grad():
        q_values = model.q_net(obs_tensor)[0].cpu().numpy()

    best_action = int(q_values.argmax())

    print(f"\nSeed {seed}")
    print(f"Best action: {ACTION_NAMES[best_action]}")

    for action, q in enumerate(q_values):
        print(
            f"  {ACTION_NAMES[action]:>8}: "
            f"{q: .5f}"
        )
