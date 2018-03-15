#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gym
from gym import spaces

from .nesenv import SCREEN_HEIGHT, SCREEN_WIDTH, NB_COLORS_SCREEN, NB_COLORS_OBS

# RESHAPED_WIDTH, RESHAPED_HEIGHT = SCREEN_HEIGHT, SCREEN_WIDTH
# RESHAPED_WIDTH, RESHAPED_HEIGHT = 110, 84
# CROPPED_WIDTH, CROPPED_HEIGHT = SCREEN_HEIGHT, SCREEN_WIDTH
# CROPPED_WIDTH, CROPPED_HEIGHT = 89, 84

# # XXX I tried to reduce the size of the observation as much as possible!
RESHAPED_WIDTH, RESHAPED_HEIGHT = 73, 64
CROPPED_WIDTH, CROPPED_HEIGHT = 60, 64

assert CROPPED_WIDTH <= RESHAPED_WIDTH, "Error: invalid value of CROPPED_WIDTH = {} > RESHAPED_WIDTH = {}...".format(CROPPED_WIDTH, RESHAPED_WIDTH)  # DEBUG
assert CROPPED_HEIGHT <= RESHAPED_HEIGHT, "Error: invalid value of CROPPED_HEIGHT = {} > RESHAPED_HEIGHT = {}...".format(CROPPED_HEIGHT, RESHAPED_HEIGHT)  # DEBUG


# XXX Change here if you want to debug and show each imframe
debug_imshow_each_frame = True
debug_imshow_each_frame = False
if debug_imshow_each_frame:
    plt.interactive(True)


# frame_skip = 1
frame_skip = 2
# frame_skip = 4
# frame_skip = 8
# frame_skip = 10


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=frame_skip):
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


def _process_frame84(frame, self=None, show=False):
    NB_COLORS = int(np.size(frame) // (SCREEN_HEIGHT * SCREEN_WIDTH))
    if NB_COLORS > 1:
        img = np.reshape(frame, [SCREEN_HEIGHT, SCREEN_WIDTH, NB_COLORS]).astype(np.float32)
    else:
        img = np.reshape(frame, [SCREEN_HEIGHT, SCREEN_WIDTH]).astype(np.float32)

    # # DEBUG by showing the *raw screen*
    # if show:
    #     print("Showing image of shape:", np.shape(img))  # DEBUG
    #     if self is not None:
    #         if self._imshow_obj is None:
    #             self._imshow_obj = plt.imshow(img) #, cmap="gray")
    #         else:
    #             self._imshow_obj.set_data(img)
    #     else:
    #         plt.imshow(img) #, cmap="gray")
    #     plt.show(block=False)
    #     plt.draw()
    #     print(input("[Close plot and enter to continue]"))  # DEBUG

    # I benchmarked, and cv2.resize is faster than skimage.transform.resize (by about 25%)
    if NB_COLORS == 3:
        resized_screen = cv2.resize(
              img[:, :, 0] * 0.299
            + img[:, :, 1] * 0.587
            + img[:, :, 2] * 0.114,
            (RESHAPED_HEIGHT, RESHAPED_WIDTH),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        resized_screen = cv2.resize(
            img,
            (RESHAPED_HEIGHT, RESHAPED_WIDTH),
            interpolation=cv2.INTER_LINEAR
        )

    resized_screen = resized_screen[8:-5, :]

    # DEBUG by showing the *observation* (to check the cropping)
    if show:
        print("Showing image of shape:", np.shape(resized_screen))  # DEBUG
        if self is not None:
            if self._imshow_obj is None:
                self._imshow_obj = plt.imshow(resized_screen, cmap="gray")
            else:
                self._imshow_obj.set_data(resized_screen)
        else:
            plt.imshow(resized_screen, cmap="gray")
        plt.show(block=False)
        plt.draw()
        print(input("[Close plot and enter to continue]"))  # DEBUG

    reshaped_screen = np.reshape(resized_screen, [CROPPED_WIDTH, CROPPED_HEIGHT, 1]).astype(np.uint8)
    return reshaped_screen


class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(RESHAPED_HEIGHT, RESHAPED_WIDTH, 1),
            dtype=np.uint8
        )
        self._imshow_obj = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs, self=self, show=debug_imshow_each_frame), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset(), self=self, show=False)


def wrap_nes_env(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env)
    return env
