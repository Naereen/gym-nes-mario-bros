import os
from gym import spaces
from .nesenv import NESEnv

package_directory = os.path.dirname(os.path.abspath(__file__))

class MarioBrosEnv(NESEnv):
    def __init__(self):
        super().__init__()
        self.lua_interface_path = os.path.join(package_directory, '../lua/mario_bros.lua')
        self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros.nes')

        self.actions = [
            'A',        # jump
            'L', 'LA',  # left, left+jump
            'R', 'RA',  # right, right+jump
            # 'B',  # run
            # 'BA', 'BL', 'BR'  # run+jump, run+left, run+right
        ]
        self.action_space = spaces.Discrete(len(self.actions))
