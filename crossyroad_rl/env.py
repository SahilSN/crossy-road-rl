import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CrossyRoadEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # 0 = wait
        # 1 = forward
        # 2 = backward
        # 3 = left
        # 4 = right
        self.action_space = spaces.Discrete(5)

        # [player_x, player_y,
        #  car1_x, car1_speed,
        #  car2_x, car2_speed]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        self.grid_width = 7
        self.goal_y = 5
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = 3
        self.player_y = 0
        self.max_y = 0
        self.steps = 0

        self.car1_x = 0.0
        self.car1_speed = 0.5

        self.car2_x = 6.0
        self.car2_speed = -0.4

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        self._move_player(action)
        self._update_cars()

        reward = -0.01

        if self.player_y > self.max_y:
            self.max_y = self.player_y
            reward += 1.0

        collision = self._check_collision()
        success = self.player_y >= self.goal_y

        terminated = collision or success
        truncated = self.steps >= self.max_steps

        if collision:
            reward -= 1.0

        if success:
            reward += 5.0

        info = {
            "max_y": self.max_y,
            "success": success,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_observation(self):
        return np.array(
            [
                self.player_x,
                self.player_y,
                self.car1_x,
                self.car1_speed,
                self.car2_x,
                self.car2_speed,
            ],
            dtype=np.float32,
        )

    def _move_player(self, action):
        if action == 1:
            self.player_y += 1
        elif action == 2:
            self.player_y -= 1
        elif action == 3:
            self.player_x -= 1
        elif action == 4:
            self.player_x += 1

        self.player_x = np.clip(
            self.player_x,
            0,
            self.grid_width - 1,
        )

        self.player_y = max(self.player_y, 0)

    def _update_cars(self):
        self.car1_x += self.car1_speed
        self.car2_x += self.car2_speed

        self.car1_x %= self.grid_width
        self.car2_x %= self.grid_width

    def _check_collision(self):
        # Road lane 1 is y = 1
        if self.player_y == 1:
            if abs(self.player_x - self.car1_x) < 0.5:
                return True

        # Road lane 2 is y = 3
        if self.player_y == 3:
            if abs(self.player_x - self.car2_x) < 0.5:
                return True

        return False