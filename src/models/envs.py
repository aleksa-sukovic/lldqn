import numpy as np

from gym.spaces import Box
from gym_anytrading.envs import StocksEnv


class EvaluateStocksEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
