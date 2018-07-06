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

delta_reward_by_life  = 1000
delta_reward_by_level = 10000

class MarioBrosEnv(NESEnv):
    def __init__(self):
        super().__init__()
        # Configuration for difference of rewards when you lose a life or win a level
        self.delta_reward_by_life = delta_reward_by_life
        self.delta_reward_by_level = delta_reward_by_level
        # Configure paths
        self.lua_interface_path = os.path.join(package_directory, '../lua/mario_bros.lua')
        self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros.nes')
        # and actions
        # TODO find the best (smallest) set of action
        self.actions = [
            'A',    # jump
            # 'L',    # left
            'LA',   # left+jump
            # 'R',    # right
            'RA',   # right+jump
            # 'B',    # run  XXX does nothing! Is it good to have an action that does nothing?
            # 'BA',   # run+jump
            'BL',   # run+left
            'BR',   # run+right
            # 'BLA',   # run+left+jump
            # 'BRA',   # run+right+jump
            # 'S',    # enter  XXX pause! Is it good to have an action that does nothing?
        ]
        # # FIXME use this one to let the dqn do nothing but run
        # self.actions = [
        #     'B'
        # ]
        self.action_space = spaces.Discrete(len(self.actions))
