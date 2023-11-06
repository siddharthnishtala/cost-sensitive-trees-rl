from gym.spaces import Box
from gym import Wrapper

import numpy as np


class FlattenObservation(Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.env = env

        state = env.reset()
        state_dimension = state.flatten().shape[0]

        self.env.observation_space = Box(-np.inf, np.inf, (state_dimension,), dtype=np.float32)

    def reset(self, **kwargs):

        observation = self.env.reset(**kwargs)

        return self.observation(observation)

    def step(self, action):

        observation, reward, done, info = self.env.step(action)

        return self.observation(observation), reward, done, info

    def observation(self, observation):
        
        return observation.flatten()
    