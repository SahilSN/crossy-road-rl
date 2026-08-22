import numpy as np
from gymnasium import spaces

from crossyroad_rl.env_v4 import CrossyRoadEnvV4


class CrossyRoadEnvV7(CrossyRoadEnvV4):
    """
    Crossy Road RL - Environment v7

    Procedural road-layout benchmark.

    Core dynamics remain based on v4:
      - grid width = 9
      - goal_y = 9
      - max_steps = 100
      - four road lanes
      - same traffic physics
      - same collision logic
      - same reward structure

    Difference:
      Road rows are randomized at every reset.

    Observation:
      local2 / egocentric observation:
        row -1
        row  0
        row +1
        row +2

      Each row contains:
        is_road
        up to 4 car slots:
            car_x
            car_speed
            active
    """

    def __init__(self):
        super().__init__()

        self.visible_row_offsets = (-1, 0, 1, 2)

        # Candidate road locations.
        # Start row y=0 remains safe.
        # Goal row y=9 remains safe.
        self.candidate_road_rows = tuple(
            range(1, self.goal_y)
        )

        # Progressive road difficulty.
        # Difficulty is assigned according to the order
        # in which roads are encountered, NOT absolute y.
        self.procedural_lane_configs = [
            {
                "min_cars": 2,
                "max_cars": 2,
                "speed_min": 0.45,
                "speed_max": 0.70,
            },
            {
                "min_cars": 3,
                "max_cars": 3,
                "speed_min": 0.55,
                "speed_max": 0.85,
            },
            {
                "min_cars": 3,
                "max_cars": 4,
                "speed_min": 0.65,
                "speed_max": 1.00,
            },
            {
                "min_cars": 4,
                "max_cars": 4,
                "speed_min": 0.75,
                "speed_max": 1.10,
            },
        ]

        # Same local2 representation as v5.
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
            float(self.grid_width - 1),
            float(self.goal_y),
            1.0,
        ]

        for _ in self.visible_row_offsets:
            # is_road
            low.append(0.0)
            high.append(1.0)

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

    def _sample_road_rows(self):
        """
        Sample exactly four road rows subject to:

          - rows are between start and goal
          - no run of >2 consecutive road rows
          - at least one road in rows 1..4
          - at least one road in rows 5..8
        """

        while True:
            rows = sorted(
                self.np_random.choice(
                    self.candidate_road_rows,
                    size=4,
                    replace=False,
                ).tolist()
            )

            first_half = any(
                row <= 4
                for row in rows
            )

            second_half = any(
                row >= 5
                for row in rows
            )

            if not (
                first_half
                and second_half
            ):
                continue

            max_run = 1
            current_run = 1

            for i in range(
                1,
                len(rows),
            ):
                if rows[i] == rows[i - 1] + 1:
                    current_run += 1

                    max_run = max(
                        max_run,
                        current_run,
                    )
                else:
                    current_run = 1

            if max_run > 2:
                continue

            return rows

    def _initialize_procedural_lanes(self):
        """
        Rebuild traffic for the currently sampled road rows.

        Lane difficulty increases according to encounter order.
        """

        self.lanes = {}

        for lane_index, row in enumerate(
            self.road_rows
        ):
            cfg = self.procedural_lane_configs[
                lane_index
            ]

            if (
                cfg["min_cars"]
                == cfg["max_cars"]
            ):
                num_cars = cfg["min_cars"]

            else:
                num_cars = int(
                    self.np_random.integers(
                        cfg["min_cars"],
                        cfg["max_cars"] + 1,
                    )
                )

            speed_mag = float(
                self.np_random.uniform(
                    cfg["speed_min"],
                    cfg["speed_max"],
                )
            )

            # Alternate directions by encountered lane.
            direction = (
                1.0
                if lane_index % 2 == 0
                else -1.0
            )

            speed = (
                direction
                * speed_mag
            )

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

            for car_index in range(
                num_cars
            ):
                x = (
                    phase
                    + car_index * spacing
                ) % self.grid_width

                cars.append({
                    "x": float(x),
                    "speed": float(speed),
                })

            self.lanes[row] = cars

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # CrossyRoadEnvV4._generate_traffic() assumes the original
        # fixed road rows [1, 3, 5, 7]. A previous v7 episode may
        # have left self.road_rows set to a procedural layout, so
        # restore the v4 rows before calling its reset.
        self.road_rows = [1, 3, 5, 7]

        super().reset(
            seed=seed,
            options=options,
        )

        # Now replace the fixed v4 layout with a seeded procedural
        # layout and rebuild traffic for those roads.
        self.road_rows = self._sample_road_rows()

        self.player_x = self.grid_width // 2
        self.player_y = 0
        self.max_y = 0
        self.steps = 0

        self._initialize_procedural_lanes()

        return (
            self._get_observation(),
            {},
        )

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

            is_road = (
                row in self.road_rows
            )

            obs.append(
                1.0
                if is_road
                else 0.0
            )

            cars = (
                self.lanes.get(
                    row,
                    [],
                )
                if is_road
                else []
            )

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
                    obs.extend([
                        -1.0,
                        0.0,
                        0.0,
                    ])

        return np.array(
            obs,
            dtype=np.float32,
        )
