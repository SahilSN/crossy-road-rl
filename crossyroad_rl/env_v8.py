import numpy as np
from gymnasium import spaces

from crossyroad_rl.env_v4 import CrossyRoadEnvV4


class CrossyRoadEnvV8(CrossyRoadEnvV4):
    """
    Crossy Road RL - Environment v8

    Mixed-mechanics benchmark:
      - fixed road + river layout
      - local2 egocentric observation
      - roads require car avoidance
      - rivers require moving-platform support

    Layout:
      y=9  goal
      y=8  safe
      y=7  river
      y=6  safe
      y=5  river
      y=4  safe
      y=3  road
      y=2  safe
      y=1  road
      y=0  start

    Core timing remains identical to v4:
      - player acts first
      - 4 physics ticks per RL action
      - dt = 0.25
      - hazards checked every physics tick
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------
        # Mixed world
        # ------------------------------------------------------------

        self.road_rows = [1, 3]
        self.river_rows = [5, 7]

        # Fixed local2 observation:
        # one row behind, current row,
        # one row ahead, two rows ahead.
        self.visible_row_offsets = (-1, 0, 1, 2)

        # Platforms have a fixed support radius.
        #
        # A player is supported if its horizontal distance
        # from a platform center is <= this value.
        self.platform_half_width = 1.0

        # ------------------------------------------------------------
        # Observation
        # ------------------------------------------------------------
        #
        # Player:
        #   x
        #   y
        #   time_remaining
        #
        # For each visible row:
        #   row_type
        #       0 = safe
        #       1 = road
        #       2 = river
        #
        #   four object slots:
        #       object_x
        #       object_speed
        #       active
        #
        # Object means:
        #   car      on a road
        #   platform on a river
        #
        # Observation size:
        #   3 + 4 * (1 + 4*3)
        #   = 55
        #

        values_per_row = (
            1
            + self.max_cars_per_lane * 3
        )

        self.obs_size = (
            3
            + len(self.visible_row_offsets)
            * values_per_row
        )

        low = [
            0.0,
            0.0,
            0.0,
        ]

        high = [
            float(self.grid_width),
            float(self.goal_y),
            1.0,
        ]

        for _ in self.visible_row_offsets:
            # row_type
            low.append(0.0)
            high.append(2.0)

            for _ in range(
                self.max_cars_per_lane
            ):
                low.extend([
                    -1.0,   # object x / inactive
                    -1.5,   # object speed
                    0.0,    # active
                ])

                high.extend([
                    float(self.grid_width),
                    1.5,
                    1.0,
                ])

        self.observation_space = spaces.Box(
            low=np.array(
                low,
                dtype=np.float32,
            ),
            high=np.array(
                high,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Separate road cars and river platforms.
        self.lanes = {}
        self.rivers = {}

    # ================================================================
    # Gymnasium API
    # ================================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):
        # Do not call v4.reset(), because v4 traffic generation assumes
        # four fixed road rows [1, 3, 5, 7].
        #
        # We only need Gymnasium's base reset to initialize np_random.
        super(CrossyRoadEnvV4, self).reset(
            seed=seed
        )

        self.player_x = float(
            self.grid_width // 2
        )
        self.player_y = 0

        self.steps = 0
        self.max_y = 0

        self._generate_world()

        observation = self._get_observation()

        info = {
            "max_y": self.max_y,
            "success": False,
            "collision": False,
            "road_collision": False,
            "drowned": False,
            "failure_type": None,
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
                self.player_x - 1.0,
                0.0,
            )

        elif action == 4:
            # right
            self.player_x = min(
                self.player_x + 1.0,
                float(self.grid_width - 1),
            )

        # ------------------------------------------------------------
        # Base reward
        # ------------------------------------------------------------

        reward = -0.01

        if self.player_y > self.max_y:
            reward += 1.0
            self.max_y = self.player_y

        terminated = False
        truncated = False

        road_collision = False
        drowned = False
        success = False

        # ------------------------------------------------------------
        # Physics
        # ------------------------------------------------------------
        #
        # Important ordering for river rows:
        #
        # 1. Determine whether the player currently has support.
        # 2. Update all moving objects.
        # 3. Carry the player with the platform it was standing on.
        # 4. Check world boundary.
        # 5. Check hazards again.
        #
        # This happens on every internal physics tick.
        #

        for _ in range(
            self.physics_ticks_per_action
        ):
            supporting_platform = None

            if self.player_y in self.river_rows:
                supporting_platform = (
                    self._get_supporting_platform()
                )

                # Entered or moved on a river row without support.
                if supporting_platform is None:
                    drowned = True
                    terminated = True
                    reward -= 1.0
                    break

            self._update_world()

            # A river platform carries the player by the same
            # displacement applied during this physics tick.
            if supporting_platform is not None:
                self.player_x += (
                    supporting_platform["speed"]
                    * self.physics_dt
                )

                # Unlike traffic/platform objects, the player does not
                # wrap around the map.
                if (
                    self.player_x < 0.0
                    or self.player_x
                    >= float(self.grid_width)
                ):
                    drowned = True
                    terminated = True
                    reward -= 1.0
                    break

                # Defensive support check after movement.
                #
                # Normally the same platform remains beneath the player
                # because they moved by exactly the same displacement.
                # This also catches boundary/wrap edge cases.
                if (
                    self._get_supporting_platform()
                    is None
                ):
                    drowned = True
                    terminated = True
                    reward -= 1.0
                    break

            if self._check_road_collision():
                road_collision = True
                terminated = True
                reward -= 1.0
                break

        # ------------------------------------------------------------
        # Goal
        # ------------------------------------------------------------

        if (
            not road_collision
            and not drowned
            and self.player_y >= self.goal_y
        ):
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

        # Compatibility:
        #
        # Existing evaluator treats "collision" as terminal hazard.
        # For v8 this includes either a car collision or drowning.
        hazard_failure = (
            road_collision
            or drowned
        )

        if road_collision:
            failure_type = "road_collision"
        elif drowned:
            failure_type = "drowned"
        else:
            failure_type = None

        info = {
            "max_y": self.max_y,
            "success": success,

            "collision": hazard_failure,

            "road_collision": road_collision,
            "drowned": drowned,
            "failure_type": failure_type,
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

    def _generate_world(self):
        self._generate_roads()
        self._generate_rivers()

    def _generate_roads(self):
        self.lanes = {}

        road_configs = {
            1: {
                "num_cars": 2,
                "min_speed": 0.45,
                "max_speed": 0.70,
                "direction": 1.0,
            },

            3: {
                "num_cars": 3,
                "min_speed": 0.55,
                "max_speed": 0.85,
                "direction": -1.0,
            },
        }

        for row in self.road_rows:
            config = road_configs[row]

            speed_magnitude = float(
                self.np_random.uniform(
                    config["min_speed"],
                    config["max_speed"],
                )
            )

            lane_speed = (
                config["direction"]
                * speed_magnitude
            )

            num_cars = config["num_cars"]

            spacing = (
                self.grid_width
                / num_cars
            )

            phase = float(
                self.np_random.uniform(
                    0.0,
                    spacing,
                )
            )

            cars = []

            for i in range(num_cars):
                car_x = (
                    phase
                    + i * spacing
                ) % self.grid_width

                cars.append({
                    "x": float(car_x),
                    "speed": float(lane_speed),
                })

            self.lanes[row] = cars

    def _generate_rivers(self):
        self.rivers = {}

        river_configs = {
            5: {
                "num_platforms": 3,
                "min_speed": 0.40,
                "max_speed": 0.60,
                "direction": 1.0,
            },

            7: {
                "num_platforms": 3,
                "min_speed": 0.55,
                "max_speed": 0.75,
                "direction": -1.0,
            },
        }

        for row in self.river_rows:
            config = river_configs[row]

            speed_magnitude = float(
                self.np_random.uniform(
                    config["min_speed"],
                    config["max_speed"],
                )
            )

            platform_speed = (
                config["direction"]
                * speed_magnitude
            )

            num_platforms = (
                config["num_platforms"]
            )

            spacing = (
                self.grid_width
                / num_platforms
            )

            phase = float(
                self.np_random.uniform(
                    0.0,
                    spacing,
                )
            )

            platforms = []

            for i in range(num_platforms):
                platform_x = (
                    phase
                    + i * spacing
                ) % self.grid_width

                platforms.append({
                    "x": float(platform_x),
                    "speed": float(
                        platform_speed
                    ),
                })

            self.rivers[row] = platforms

    # ================================================================
    # Physics
    # ================================================================

    def _update_world(self):
        # Road cars.
        for row in self.road_rows:
            for car in self.lanes[row]:
                car["x"] += (
                    car["speed"]
                    * self.physics_dt
                )

                car["x"] %= self.grid_width

        # River platforms.
        for row in self.river_rows:
            for platform in self.rivers[row]:
                platform["x"] += (
                    platform["speed"]
                    * self.physics_dt
                )

                platform["x"] %= self.grid_width

    def _check_road_collision(self):
        if self.player_y not in self.road_rows:
            return False

        for car in self.lanes[
            self.player_y
        ]:
            distance = self._circular_distance(
                self.player_x,
                car["x"],
            )

            if distance < self.collision_distance:
                return True

        return False

    def _get_supporting_platform(self):
        if self.player_y not in self.river_rows:
            return None

        for platform in self.rivers[
            self.player_y
        ]:
            distance = self._circular_distance(
                self.player_x,
                platform["x"],
            )

            if distance <= self.platform_half_width:
                return platform

        return None

    # ================================================================
    # Observation
    # ================================================================

    def _get_row_type(self, row):
        if row in self.road_rows:
            return 1.0

        if row in self.river_rows:
            return 2.0

        return 0.0

    def _get_row_objects(self, row):
        if row in self.road_rows:
            return self.lanes.get(
                row,
                [],
            )

        if row in self.river_rows:
            return self.rivers.get(
                row,
                [],
            )

        return []

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

        for offset in self.visible_row_offsets:
            row = (
                self.player_y
                + offset
            )

            row_type = (
                self._get_row_type(row)
            )

            obs.append(row_type)

            objects = (
                self._get_row_objects(row)
            )

            for slot in range(
                self.max_cars_per_lane
            ):
                if slot < len(objects):
                    obj = objects[slot]

                    obs.extend([
                        float(obj["x"]),
                        float(obj["speed"]),
                        1.0,
                    ])

                else:
                    obs.extend([
                        -1.0,
                        0.0,
                        0.0,
                    ])

        return np.array(
            obs,
            dtype=np.float32,
        )
