import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


MAX_STATE_SPACE_DIM = 10


class AugmentObservationSpaceWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        high = self.observation_space.high
        high = np.pad(high, pad_width=(0, MAX_STATE_SPACE_DIM - len(high)), mode="constant")
        low = -high
        self._observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation
