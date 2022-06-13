import cv2

from collections import deque
from gym import Env, ObservationWrapper
from gym.wrappers.frame_stack import LazyFrames


class PreprocessObservation(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.screen_size: int = 84

    def observation(self, observation):
        size = (self.screen_size, self.screen_size)

        pixels = observation["pixels"]
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        pixels = pixels / 255.0

        return pixels


class StackObservation(ObservationWrapper):
    def __init__(self, env: Env, num_stack: int = 4, lz4_compress: bool = False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.frames = deque(maxlen=num_stack)

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused

        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return self.observation(None), info
        else:
            return self.observation(None)
