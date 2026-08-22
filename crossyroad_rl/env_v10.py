import numpy as np

from crossyroad_rl.env_v9 import CrossyRoadEnvV9


class CrossyRoadEnvV10(CrossyRoadEnvV9):
    """
    V10: V9 with episode-level speed domain randomization.

    Everything about V9 is preserved except that a new hazard speed
    multiplier is sampled once per episode.

    Training distribution:
        speed_scale ~ Uniform(speed_scale_min, speed_scale_max)

    Default:
        Uniform(0.8, 1.2)

    The sampled scale applies to both road cars and river platforms
    through V9's existing speed_scale mechanism.
    """

    def __init__(
        self,
        speed_scale_min=0.8,
        speed_scale_max=1.2,
        composition="standard",
    ):
        if speed_scale_min <= 0:
            raise ValueError(
                "speed_scale_min must be > 0"
            )

        if speed_scale_max <= 0:
            raise ValueError(
                "speed_scale_max must be > 0"
            )

        if speed_scale_min > speed_scale_max:
            raise ValueError(
                "speed_scale_min must be <= speed_scale_max"
            )

        self.speed_scale_min = float(
            speed_scale_min
        )

        self.speed_scale_max = float(
            speed_scale_max
        )

        # Separate RNG for episode-level speed randomization.
        #
        # It is intentionally separate from V9's procedural-layout RNG,
        # so introducing speed randomization does not consume random
        # numbers from the layout-generation stream.
        self._speed_rng = np.random.default_rng()

        self.current_speed_scale = 1.0

        super().__init__(
            speed_scale=1.0,
            composition=composition,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # If Gym/SB3 supplies a seed, deterministically initialize
        # the speed-randomization stream as well.
        #
        # The offset keeps this stream separate from the environment's
        # own procedural RNG stream.
        if seed is not None:
            self._speed_rng = np.random.default_rng(
                int(seed) + 1_000_003
            )

        self.current_speed_scale = float(
            self._speed_rng.uniform(
                self.speed_scale_min,
                self.speed_scale_max,
            )
        )

        # V9 already multiplies generated road/platform speeds by
        # self.speed_scale, so set the sampled value before reset.
        self.speed_scale = self.current_speed_scale

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        # Make the sampled domain parameter observable in logs/tests.
        info = dict(info)
        info["speed_scale"] = (
            self.current_speed_scale
        )

        return obs, info
