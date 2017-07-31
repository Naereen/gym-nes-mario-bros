import gym
import nesgym

print('env:', nesgym.NESEnv)

env = gym.make('nesgym/NESEnv-v0')
env.reset()

for step in range(100):
    env.render()
    obs, reward, done, info = env.step(0)
