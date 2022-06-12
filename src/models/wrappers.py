import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper, ActionWrapper


MAX_STATE_SPACE_DIM = 10


class AugmentObservationSpaceWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        high = self.observation_space.high
        high = np.pad(high, pad_width=(0, MAX_STATE_SPACE_DIM - len(high)), mode="constant")
        low = -high
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        padding = (0, MAX_STATE_SPACE_DIM - len(observation))
        return np.pad(observation, pad_width=padding, mode="constant")


class TestWrapper(ActionWrapper):
    def action(self, act):
        if not isinstance(act, np.ndarray):
            act = np.array([act])
        return act
