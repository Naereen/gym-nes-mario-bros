import os
import gym
from gym import wrappers
import nesgym
import numpy as np

env = gym.make('nesgym/NekketsuSoccerPK-v0')
env = nesgym.wrap_nes_env(env)
expt_dir = '/tmp/soccer/'
env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True)

env.reset()

for step in range(10000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if reward != 0:
        print('reward:', reward)
    if done:
        env.reset()

env.close()
