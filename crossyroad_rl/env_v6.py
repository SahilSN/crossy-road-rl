from gymnasium import spaces
import numpy as np

from crossyroad_rl.env_v4 import CrossyRoadEnvV4


class CrossyRoadEnvV6(CrossyRoadEnvV4):
    """
    Crossy Road RL - Environment v6

    Observation-horizon experiment.

    World dynamics are identical to v4:
      - same grid
      - same traffic
      - same rewards
      - same physics
      - same episode length

    The only manipulated variable is how many rows ahead
    of the player are visible.

    Local observation always includes:
      - one row behind
      - current row
      - N rows ahead

    Example:
      horizon=1 -> offsets (-1, 0, 1)
      horizon=2 -> offsets (-1, 0, 1, 2)
      horizon=3 -> offsets (-1, 0, 1, 2, 3)
    """

    def __init__(self, observation_horizon=2):
        super().__init__()

        if observation_horizon not in (1, 2, 3):
            raise ValueError(
                "observation_horizon must be 1, 2, or 3"
            )

        self.observation_horizon = observation_horizon

        self.visible_row_offsets = tuple(
            range(-1, observation_horizon + 1)
        )

        # Player:
        #   x, y, time_remaining
        #
        # Each visible row:
        #   is_road
        #   4 * (car_x, car_speed, active)
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
            float(self.grid_width - 1),
            float(self.goal_y),
            1.0,
        ]

        for _ in self.visible_row_offsets:
            # is_road
            low.append(0.0)
            high.append(1.0)

            for _ in range(self.max_cars_per_lane):
                low.extend([
                    -1.0,   # car_x
                    -1.5,   # car_speed
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
            row = self.player_y + offset

            is_road = row in self.road_rows

            obs.append(
                1.0 if is_road else 0.0
            )

            if is_road:
                cars = self.lanes[row]
            else:
                cars = []

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
