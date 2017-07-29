# test script
import subprocess

args = ['fceux', '--loadlua', 'lua/nes_interface.lua', '../roms/1.nes']
proc = subprocess.Popen(' '.join(args), shell=True)
proc.communicate()
