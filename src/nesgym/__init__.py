#!/usr/bin/env python
# -*- coding: utf-8 -*-
# By Lilian Besson (Naereen)
# https://github.com/Naereen/gym-nes-mario-bros
# MIT License https://lbesson.mit-license.org/
#
from __future__ import division, print_function  # Python 2 compatibility

import os
from gym.envs.registration import register
from .nesenv import NESEnv
from .nekketsu_soccer_env import NekketsuSoccerPKEnv
from .mario_bros_env import MarioBrosEnv
from .wrappers import wrap_nes_env

register(
    id='nesgym/NekketsuSoccerPK-v0',
    entry_point='nesgym:NekketsuSoccerPKEnv',
    max_episode_steps=9999999,
    reward_threshold=32000,
    kwargs={},
    nondeterministic=True,
)

register(
    id='nesgym/MarioBros-v0',
    entry_point='nesgym:MarioBrosEnv',
    max_episode_steps=9999999,
    reward_threshold=32000,
    kwargs={},
    nondeterministic=True,
)
