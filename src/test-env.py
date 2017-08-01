import gym
import nesgym
import numpy as np

print('env:', nesgym.NESEnv)

env = gym.make('nesgym/NESEnv-v0')
env.reset()

for step in range(10000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
