import os
from gym.envs.registration import register
from .nesenv import NESEnv

register(
    id='nesgym/NESEnv-v0',
    entry_point='nesgym:NESEnv',
    max_episode_steps=9999999,
    reward_threshold=32000,
    kwargs={},
    nondeterministic=True,
)
