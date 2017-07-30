#!/usr/bin/python

import subprocess
import os
import sys

from threading import Thread

package_directory = os.path.dirname(os.path.abspath(__file__))
SEP = '|'

def ensure_create_pipe(pipe_name):
    if not os.path.exists(pipe_name):
        os.mkfifo(pipe_name)

def pipe_listen(pipe_name):
    with open(pipe_name, 'rb') as pipe:
        while True:
            msg = pipe.readline()
            #print('message: ', msg[:100])
            body = msg.split(b'\xFF')
            msg_type, frame = body[0], body[1]
            msg_type = msg_type.decode('ascii')
            frame = int(frame.decode('ascii'))
            #print('incoming message:', msg_type, frame, file=sys.stderr)


pipe_out = None
def write_to_pipe(message):
    global pipe_out
    if not pipe_out:
        # 1 - line buffering
        pipe_out = open(pipe_out_name, 'w', 1)
    pipe_out.write(message + '\n')
    pipe_out.flush()

def reset():
    write_to_pipe('reset' + SEP)

def joypad(button):
    write_to_pipe('joypad' + SEP + button)

# emulator to client
pipe_in_name = '/tmp/nesgym-pipe-in'
# client to emulator
pipe_out_name = '/tmp/nesgym-pipe-out'

ensure_create_pipe(pipe_in_name)
ensure_create_pipe(pipe_out_name)

thread_incoming = Thread(target=pipe_listen, kwargs={'pipe_name': pipe_in_name})
thread_incoming.start()

args = ['fceux', '--loadlua', os.path.join(package_directory, 'lua/soccer.lua'), '../roms/soccer.nes', '&']
proc = subprocess.Popen(' '.join(args), shell=True)
print('started proc')
proc.communicate()

print('reset')
reset()

for step in range(1000):
    #joypad("RB")
    pass

print('end step')
thread_incoming.join()
