#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

from collections import deque
import cv2
import numpy as np
import gym
from gym import spaces

from .nesenv import SCREEN_HEIGHT, SCREEN_WIDTH
RESHAPED_HEIGHT, RESHAPED_WIDTH = 84, 110

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


def _process_frame84(frame):
    img = np.reshape(frame, [SCREEN_HEIGHT, SCREEN_WIDTH, 3]).astype(np.float32)
    # I benchmarked, and cv2.resize is faster than skimage.transform.resize (by about 25%)
    resized_screen = cv2.resize(
          img[:, :, 0] * 0.299
        + img[:, :, 1] * 0.587
        + img[:, :, 2] * 0.114,
        (RESHAPED_HEIGHT, RESHAPED_WIDTH),
        interpolation=cv2.INTER_LINEAR
    )
    return np.reshape(resized_screen, [RESHAPED_HEIGHT, RESHAPED_WIDTH, 1]).astype(np.uint8)


class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(RESHAPED_HEIGHT, RESHAPED_WIDTH, 1),
            dtype=np.uint8
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())


def wrap_nes_env(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    return env
