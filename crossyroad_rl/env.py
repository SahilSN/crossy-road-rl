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

        self.grid_width = 7
        self.goal_y = 5
        self.max_steps = 200

        # Each agent action advances the simulation through several
        # smaller physics steps.
        self.physics_ticks_per_action = 4
        self.physics_dt = 0.25

        self.cars_per_lane = 3

        # Observation:
        # player_x, player_y,
        # then x + speed for each car on both road lanes.
        obs_size = 2 + (2 * self.cars_per_lane * 2)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.road_lanes = {
            1: [],
            3: [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = 3.0
        self.player_y = 0

        self.max_y = 0
        self.steps = 0

        self._generate_cars()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        self._move_player(action)

        collision = False

        # Traffic keeps moving during the action.
        for _ in range(self.physics_ticks_per_action):
            self._update_cars()

            if self._check_collision():
                collision = True
                break

        reward = -0.01

        if self.player_y > self.max_y:
            self.max_y = self.player_y
            reward += 1.0

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
            "collision": collision,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _generate_cars(self):
        self.road_lanes = {
            1: [],
            3: [],
        }

        # One lane moves right, one moves left.
        lane1_speed = self.np_random.uniform(0.7, 1.2)
        lane2_speed = -self.np_random.uniform(0.7, 1.2)

        for lane_y, speed in [
            (1, lane1_speed),
            (3, lane2_speed),
        ]:
            # Random offset means every reset has a different traffic phase.
            offset = self.np_random.uniform(
                0.0,
                self.grid_width,
            )

            spacing = self.grid_width / self.cars_per_lane

            for i in range(self.cars_per_lane):
                car_x = (
                    offset + i * spacing
                ) % self.grid_width

                self.road_lanes[lane_y].append(
                    {
                        "x": float(car_x),
                        "speed": float(speed),
                    }
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

        self.player_x = float(
            np.clip(
                self.player_x,
                0,
                self.grid_width - 1,
            )
        )

        self.player_y = max(
            self.player_y,
            0,
        )

    def _update_cars(self):
        for cars in self.road_lanes.values():
            for car in cars:
                car["x"] += (
                    car["speed"]
                    * self.physics_dt
                )

                car["x"] %= self.grid_width

    def _circular_distance(self, x1, x2):
        direct = abs(x1 - x2)
        wrapped = self.grid_width - direct

        return min(
            direct,
            wrapped,
        )

    def _check_collision(self):
        if self.player_y not in self.road_lanes:
            return False

        for car in self.road_lanes[self.player_y]:
            distance = self._circular_distance(
                self.player_x,
                car["x"],
            )

            if distance < 0.45:
                return True

        return False

    def _get_observation(self):
        obs = [
            self.player_x,
            float(self.player_y),
        ]

        for lane_y in [1, 3]:
            for car in self.road_lanes[lane_y]:
                obs.extend(
                    [
                        car["x"],
                        car["speed"],
                    ]
                )

        return np.array(
            obs,
            dtype=np.float32,
        )
