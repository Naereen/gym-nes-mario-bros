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

class NekketsuSoccerPKEnv(NESEnv):
    def __init__(self):
        super().__init__()
        self.lua_interface_path = os.path.join(package_directory, '../lua/soccer.lua')
        self.rom_file_path = os.path.join(package_directory, '../roms/soccer.nes')
        self.actions = [
            'R', 'UR', 'DR',
            'B', 'URB', 'DRB', 'RB',
            'AB', 'RAB', 'URAB', 'DRAB'
        ]
        self.action_space = spaces.Discrete(len(self.actions))
