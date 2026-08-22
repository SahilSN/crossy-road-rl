import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crossyroad_rl.env_v8 import CrossyRoadEnvV8


class CrossyRoadEnvV9(CrossyRoadEnvV8):
    """
    Crossy Road RL - Environment v9

    Procedural mixed-mechanics benchmark.

    Each episode:
      - sample 4 hazard rows from y=1..8
      - assign each hazard as road or river
      - require at least one of each
      - no more than 2 consecutive hazards
      - difficulty increases with hazard encounter order

    Observation remains local2:
      row -1
      row  0
      row +1
      row +2

    row_type:
      0 = safe
      1 = road
      2 = river
    """

    def __init__(self, speed_scale=1.0):
        super().__init__()

        self.speed_scale = float(speed_scale)

        if self.speed_scale <= 0:
            raise ValueError('speed_scale must be > 0')

        self.candidate_hazard_rows = tuple(
            range(1, self.goal_y)
        )

        # ------------------------------------------------------------
        # Difficulty schedules
        # ------------------------------------------------------------

        # Difficulty is indexed by hazard encounter order:
        # first, second, third, fourth.

        self.road_difficulty = [
            {
                "min_objects": 2,
                "max_objects": 2,
                "min_speed": 0.45,
                "max_speed": 0.70,
            },
            {
                "min_objects": 2,
                "max_objects": 3,
                "min_speed": 0.50,
                "max_speed": 0.80,
            },
            {
                "min_objects": 3,
                "max_objects": 3,
                "min_speed": 0.60,
                "max_speed": 0.95,
            },
            {
                "min_objects": 3,
                "max_objects": 4,
                "min_speed": 0.70,
                "max_speed": 1.10,
            },
        ]

        self.river_difficulty = [
            {
                "min_objects": 3,
                "max_objects": 3,
                "min_speed": 0.35,
                "max_speed": 0.50,
                "half_width": 1.10,
            },
            {
                "min_objects": 3,
                "max_objects": 3,
                "min_speed": 0.40,
                "max_speed": 0.60,
                "half_width": 1.05,
            },
            {
                "min_objects": 3,
                "max_objects": 3,
                "min_speed": 0.50,
                "max_speed": 0.70,
                "half_width": 1.00,
            },
            {
                "min_objects": 3,
                "max_objects": 3,
                "min_speed": 0.60,
                "max_speed": 0.80,
                "half_width": 0.95,
            },
        ]

        # Per-river support widths.
        self.river_half_widths = {}

        # Current sampled layout.
        self.hazard_rows = []
        self.hazard_types = {}

        # v8 already has the correct 55-dimensional observation
        # format, but rebuild explicitly for clarity.
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
            low.append(0.0)
            high.append(2.0)

            for _ in range(
                self.max_cars_per_lane
            ):
                low.extend([
                    -1.0,
                    -1.5,
                    0.0,
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

    # ================================================================
    # Reset
    # ================================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):
        # Initialize Gymnasium's RNG directly.
        #
        # Do not call any inherited CrossyRoad reset method here:
        # v4/v8 assume fixed lane layouts during their reset logic,
        # while v9 samples arbitrary procedural hazard rows.
        gym.Env.reset(
            self,
            seed=seed,
        )

        self.player_x = float(
            self.grid_width // 2
        )
        self.player_y = 0

        self.steps = 0
        self.max_y = 0

        self._sample_layout()
        self._generate_procedural_world()

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

    # ================================================================
    # Procedural layout
    # ================================================================

    def _sample_layout(self):
        while True:
            rows = sorted(
                self.np_random.choice(
                    self.candidate_hazard_rows,
                    size=4,
                    replace=False,
                ).tolist()
            )

            # At least one hazard in each half.
            if not any(
                row <= 4
                for row in rows
            ):
                continue

            if not any(
                row >= 5
                for row in rows
            ):
                continue

            # No more than 2 consecutive hazards.
            max_run = 1
            current_run = 1

            for i in range(
                1,
                len(rows),
            ):
                if (
                    rows[i]
                    == rows[i - 1] + 1
                ):
                    current_run += 1

                    max_run = max(
                        max_run,
                        current_run,
                    )

                else:
                    current_run = 1

            if max_run > 2:
                continue

            break

        # Assign road/river labels until both mechanics appear.
        while True:
            types = self.np_random.choice(
                ["road", "river"],
                size=4,
                replace=True,
            ).tolist()

            if (
                "road" in types
                and "river" in types
            ):
                break

        self.hazard_rows = rows

        self.hazard_types = {
            row: hazard_type
            for row, hazard_type
            in zip(rows, types)
        }

        self.road_rows = [
            row
            for row in rows
            if self.hazard_types[row] == "road"
        ]

        self.river_rows = [
            row
            for row in rows
            if self.hazard_types[row] == "river"
        ]

    # ================================================================
    # World generation
    # ================================================================

    def _generate_procedural_world(self):
        self.lanes = {}
        self.rivers = {}
        self.river_half_widths = {}

        for hazard_index, row in enumerate(
            self.hazard_rows
        ):
            hazard_type = (
                self.hazard_types[row]
            )

            # Alternate movement direction by encountered hazard.
            direction = (
                1.0
                if hazard_index % 2 == 0
                else -1.0
            )

            if hazard_type == "road":
                self._generate_road(
                    row,
                    hazard_index,
                    direction,
                )

            else:
                self._generate_river(
                    row,
                    hazard_index,
                    direction,
                )

    def _generate_road(
        self,
        row,
        hazard_index,
        direction,
    ):
        cfg = self.road_difficulty[
            hazard_index
        ]

        num_objects = int(
            self.np_random.integers(
                cfg["min_objects"],
                cfg["max_objects"] + 1,
            )
        )

        speed_magnitude = float(
            self.np_random.uniform(
                cfg["min_speed"],
                cfg["max_speed"],
            )
        )

        speed = (
            direction
            * speed_magnitude
            * self.speed_scale
        )

        spacing = (
            self.grid_width
            / num_objects
        )

        phase = float(
            self.np_random.uniform(
                0.0,
                spacing,
            )
        )

        cars = []

        for i in range(num_objects):
            x = (
                phase
                + i * spacing
            ) % self.grid_width

            cars.append({
                "x": float(x),
                "speed": float(speed),
            })

        self.lanes[row] = cars

    def _generate_river(
        self,
        row,
        hazard_index,
        direction,
    ):
        cfg = self.river_difficulty[
            hazard_index
        ]

        num_objects = int(
            self.np_random.integers(
                cfg["min_objects"],
                cfg["max_objects"] + 1,
            )
        )

        speed_magnitude = float(
            self.np_random.uniform(
                cfg["min_speed"],
                cfg["max_speed"],
            )
        )

        speed = (
            direction
            * speed_magnitude
            * self.speed_scale
        )

        spacing = (
            self.grid_width
            / num_objects
        )

        phase = float(
            self.np_random.uniform(
                0.0,
                spacing,
            )
        )

        platforms = []

        for i in range(num_objects):
            x = (
                phase
                + i * spacing
            ) % self.grid_width

            platforms.append({
                "x": float(x),
                "speed": float(speed),
            })

        self.rivers[row] = platforms

        self.river_half_widths[row] = (
            cfg["half_width"]
        )

    # ================================================================
    # River support
    # ================================================================

    def _get_supporting_platform(self):
        if (
            self.player_y
            not in self.river_rows
        ):
            return None

        half_width = (
            self.river_half_widths[
                self.player_y
            ]
        )

        for platform in self.rivers[
            self.player_y
        ]:
            distance = (
                self._circular_distance(
                    self.player_x,
                    platform["x"],
                )
            )

            if distance <= half_width:
                return platform

        return None

    # ================================================================
    # Observation helpers
    # ================================================================

    def _get_row_type(self, row):
        hazard_type = (
            self.hazard_types.get(row)
        )

        if hazard_type == "road":
            return 1.0

        if hazard_type == "river":
            return 2.0

        return 0.0

    def _get_row_objects(self, row):
        hazard_type = (
            self.hazard_types.get(row)
        )

        if hazard_type == "road":
            return self.lanes.get(
                row,
                [],
            )

        if hazard_type == "river":
            return self.rivers.get(
                row,
                [],
            )

        return []
