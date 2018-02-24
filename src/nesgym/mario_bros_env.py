import os
from gym import spaces
from .nesenv import NESEnv

package_directory = os.path.dirname(os.path.abspath(__file__))

class MarioBrosEnv(NESEnv):
    def __init__(self):
        super().__init__()
        # Configure paths
        self.lua_interface_path = os.path.join(package_directory, '../lua/mario_bros.lua')
        # self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros.nes')
        self.rom_file_path = os.path.join(package_directory, '../roms/mario_bros_ju.nes')
        # and actions
        self.actions = [
            'A',    # jump
            'L',    # left
            'LA',   # left+jump
            'R',    # right
            'RA',   # right+jump
            # 'B',  # run
            # 'BA', # run+jump
            # 'BL', # run+left
            # 'BR', # run+right
        ]
        self.action_space = spaces.Discrete(len(self.actions))
