import gym
from gym import utils, spaces
from gym.utils import seeding

class NESEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self.curr_seed = 0
        self.screen = None

    def _step(self, action):
        print('step')
        obs = []
        reward = 0
        info = {}
        return obs, reward, False, info

    def _reset(self):
        print('reset env')

    def _render(self, mode='human', close=False):
        print('render')

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 256
        return [self.curr_seed]
