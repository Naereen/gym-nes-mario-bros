import os
import subprocess
import sys
from threading import Thread

import gym
from gym import utils, spaces
from gym.utils import seeding

package_directory = os.path.dirname(os.path.abspath(__file__))
SEP = '|'

class NESEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self.curr_seed = 0
        self.screen = None
        self.closed = False
        self.actions = [
            'U', 'D', 'L', 'R',
            'UR', 'DR', 'URA', 'DRB',
            'A', 'B', 'RB', 'RA']
        self.frame = 0

        self.pipe_in = None
        self.pipe_out = None
        # for communication with emulator
        self._open_pipes()

        self.thread_incoming = Thread(target=self._pipe_handler)
        self.thread_incoming.start()

        args = ['fceux', '--loadlua', os.path.join(package_directory, '../lua/soccer.lua'), '../roms/soccer.nes', '&']
        proc = subprocess.Popen(' '.join(args), shell=True)
        print('started proc')
        proc.communicate()

    ## ---------- gym.Env methods -------------
    def _step(self, action):
        self.frame += 1
        done = False
        if self.frame >= 420:
            done = True
            self.frame = 0
        obs = []
        reward = 0
        info = {}
        self._joypad(self.actions[action])
        return obs, reward, done, info

    def _reset(self):
        self._write_to_pipe('reset' + SEP)
        print('reset env')

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 256
        return [self.curr_seed]

    def _close(self):
        self.closed = True
    ## ------------- end gym.Env --------------

    ## ------------- emulator related ------------
    def _joypad(self, button):
        self._write_to_pipe('joypad' + SEP + button)
    ## ------------  end emulator  -------------

    ## ------------- pipes ---------------
    def _write_to_pipe(self, message):
        if not self.pipe_out:
            # arg 1 for line buffering - see python doc
            self.pipe_out = open(self.pipe_out_name, 'w', 1)
        self.pipe_out.write(message + '\n')
        self.pipe_out.flush()

    def _pipe_handler(self):
        with open(self.pipe_in_name, 'rb') as pipe:
            while not self.closed:
                msg = pipe.readline()
                #print('message: ', msg[:100])
                body = msg.split(b'\xFF')
                msg_type, frame = body[0], body[1]
                msg_type = msg_type.decode('ascii')
                frame = int(frame.decode('ascii'))
                print('message: ', msg_type, 'frame: ', frame)

    def _open_pipes(self):
        # emulator to client
        self.pipe_in_name = '/tmp/nesgym-pipe-in'
        # client to emulator
        self.pipe_out_name = '/tmp/nesgym-pipe-out'
        self._ensure_create_pipe(self.pipe_in_name)
        self._ensure_create_pipe(self.pipe_out_name)

    def _ensure_create_pipe(self, pipe_name):
        if not os.path.exists(pipe_name):
            os.mkfifo(pipe_name)
    ## ------------ end pipes --------------
