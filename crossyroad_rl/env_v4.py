import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CrossyRoadEnvV4(gym.Env):
    """
    Crossy Road RL - Environment v4

    Harder road-only environment:
      - 9-column grid
      - goal row 9
      - 4 independently randomized road lanes
      - 3-5 cars per lane
      - alternating traffic directions
      - independent lane speeds
      - 4 internal physics ticks per RL action
      - fully observable state
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------
        # World configuration
        # ------------------------------------------------------------

        self.grid_width = 9
        self.goal_y = 9
        self.max_steps = 100

        self.road_rows = [1, 3, 5, 7]

        self.min_cars_per_lane = 2
        self.max_cars_per_lane = 4

        # Traffic updates multiple times per agent action.
        self.physics_ticks_per_action = 4
        self.physics_dt = 0.25

        # Cars collide when their circular horizontal distance
        # from the player is less than this threshold.
        self.collision_distance = 0.5

        # ------------------------------------------------------------
        # Action space
        # ------------------------------------------------------------
        #
        # 0 = wait
        # 1 = forward
        # 2 = backward
        # 3 = left
        # 4 = right
        #
        self.action_space = spaces.Discrete(5)

        # ------------------------------------------------------------
        # Observation
        # ------------------------------------------------------------
        #
        # Player:
        #   player_x
        #   player_y
        #   time_remaining
        #
        # For each of 4 road lanes and each of 5 possible car slots:
        #   car_x
        #   car_speed
        #   active
        #
        # Observation size:
        #
        #   3 + (4 lanes * 5 cars * 3 values)
        #   = 63
        #
        self.obs_size = (
            3
            + len(self.road_rows)
            * self.max_cars_per_lane
            * 3
        )

        low = [
            0.0,    # player_x
            0.0,    # player_y
            0.0,    # time_remaining
        ]

        high = [
            float(self.grid_width - 1),
            float(self.goal_y),
            1.0,
        ]

        for _ in self.road_rows:
            for _ in range(self.max_cars_per_lane):
                low.extend([
                    -1.0,   # x; -1 indicates inactive slot
                    -1.5,   # speed
                    0.0,    # active
                ])

                high.extend([
                    float(self.grid_width),
                    1.5,
                    1.0,
                ])

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------

        self.player_x = None
        self.player_y = None

        self.steps = None
        self.max_y = None

        # Dictionary:
        #
        # {
        #   1: [
        #       {"x": ..., "speed": ...},
        #       ...
        #   ],
        #   ...
        # }
        #
        self.lanes = {}

    # ================================================================
    # Gymnasium API
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = self.grid_width // 2
        self.player_y = 0

        self.steps = 0
        self.max_y = 0

        self._generate_traffic()

        observation = self._get_observation()

        info = {
            "max_y": self.max_y,
            "success": False,
            "collision": False,
        }

        return observation, info

    def step(self, action):
        self.steps += 1

        # ------------------------------------------------------------
        # Move player
        # ------------------------------------------------------------

        if action == 0:
            # wait
            pass

        elif action == 1:
            # forward
            self.player_y = min(
                self.player_y + 1,
                self.goal_y,
            )

        elif action == 2:
            # backward
            self.player_y = max(
                self.player_y - 1,
                0,
            )

        elif action == 3:
            # left
            self.player_x = max(
                self.player_x - 1,
                0,
            )

        elif action == 4:
            # right
            self.player_x = min(
                self.player_x + 1,
                self.grid_width - 1,
            )

        # ------------------------------------------------------------
        # Base reward
        # ------------------------------------------------------------

        reward = -0.01

        # Reward newly achieved forward progress.
        if self.player_y > self.max_y:
            reward += 1.0
            self.max_y = self.player_y

        terminated = False
        truncated = False

        collision = False
        success = False

        # ------------------------------------------------------------
        # Traffic physics
        # ------------------------------------------------------------

        for _ in range(self.physics_ticks_per_action):
            self._update_traffic()

            if self._check_collision():
                collision = True
                terminated = True
                reward -= 1.0
                break

        # ------------------------------------------------------------
        # Goal
        # ------------------------------------------------------------

        if not collision and self.player_y >= self.goal_y:
            success = True
            terminated = True
            reward += 5.0

        # ------------------------------------------------------------
        # Timeout
        # ------------------------------------------------------------

        if (
            not terminated
            and self.steps >= self.max_steps
        ):
            truncated = True
            reward -= 5.0

        observation = self._get_observation()

        info = {
            "max_y": self.max_y,
            "success": success,
            "collision": collision,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ================================================================
    # World generation
    # ================================================================

    def _generate_traffic(self):
        self.lanes = {}

        lane_configs = {
            1: {
                "min_cars": 2,
                "max_cars": 2,
                "min_speed": 0.45,
                "max_speed": 0.70,
            },
            3: {
                "min_cars": 3,
                "max_cars": 3,
                "min_speed": 0.55,
                "max_speed": 0.85,
            },
            5: {
                "min_cars": 3,
                "max_cars": 4,
                "min_speed": 0.65,
                "max_speed": 1.00,
            },
            7: {
                "min_cars": 4,
                "max_cars": 4,
                "min_speed": 0.75,
                "max_speed": 1.10,
            },
        }

        for lane_index, row in enumerate(self.road_rows):

            config = lane_configs[row]

            # Alternate traffic direction by lane.
            direction = (
                1.0
                if lane_index % 2 == 0
                else -1.0
            )

            speed_magnitude = self.np_random.uniform(
                config["min_speed"],
                config["max_speed"],
            )

            lane_speed = (
                direction * speed_magnitude
            )

            num_cars = int(
                self.np_random.integers(
                    config["min_cars"],
                    config["max_cars"] + 1,
                )
            )

            spacing = (
                self.grid_width / num_cars
            )

            phase = self.np_random.uniform(
                0.0,
                spacing,
            )

            cars = []

            for i in range(num_cars):
                car_x = (
                    phase + i * spacing
                ) % self.grid_width

                cars.append({
                    "x": float(car_x),
                    "speed": float(lane_speed),
                })

            self.lanes[row] = cars

    # ================================================================
    # Physics
    # ================================================================

    def _update_traffic(self):
        for row in self.road_rows:
            for car in self.lanes[row]:

                car["x"] += (
                    car["speed"]
                    * self.physics_dt
                )

                car["x"] %= self.grid_width

    def _circular_distance(self, x1, x2):
        direct = abs(x1 - x2)

        wrapped = (
            self.grid_width - direct
        )

        return min(
            direct,
            wrapped,
        )

    def _check_collision(self):
        if self.player_y not in self.road_rows:
            return False

        cars = self.lanes[self.player_y]

        for car in cars:
            distance = self._circular_distance(
                self.player_x,
                car["x"],
            )

            if distance < self.collision_distance:
                return True

        return False

    # ================================================================
    # Observation
    # ================================================================

    def _get_observation(self):
        time_remaining = (
            1.0
            - self.steps / self.max_steps
        )

        time_remaining = max(
            0.0,
            time_remaining,
        )

        obs = [
            float(self.player_x),
            float(self.player_y),
            float(time_remaining),
        ]

        for row in self.road_rows:
            cars = self.lanes[row]

            for slot in range(
                self.max_cars_per_lane
            ):

                if slot < len(cars):
                    car = cars[slot]

                    obs.extend([
                        float(car["x"]),
                        float(car["speed"]),
                        1.0,
                    ])

                else:
                    # Empty padded slot.
                    obs.extend([
                        -1.0,
                        0.0,
                        0.0,
                    ])

        return np.array(
            obs,
            dtype=np.float32,
        )
