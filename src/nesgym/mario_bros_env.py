#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

import os
from gym import spaces
from .nesenv import NESEnv

package_directory = os.path.dirname(os.path.abspath(__file__))

class MarioBrosEnv(NESEnv):
    def __init__(self):
        super().__init__()
        # Configure paths
        self.lua_interface_path = os.path.join(package_directory, '../lua/mario_bros.lua')
        # self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros.nes')
        self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros_ju.nes')
        # and actions
        self.actions = [
            'A',    # jump
            'L',    # left
            'LA',   # left+jump
            'R',    # right
            'RA',   # right+jump
            # 'B',  # run
            # 'BA', # run+jump
            # 'BL', # run+left
            # 'BR', # run+right
        ]
        self.action_space = spaces.Discrete(len(self.actions))


    ## ---------- gym.Env methods -------------
    def _step(self, action):
        obs, self.reward, done, info = super()._step(action)
        return obs, self.reward, done or (self.life == 0), info