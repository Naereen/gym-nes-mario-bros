-- #!/usr/bin/env lua
-- By Lilian Besson (Naereen)
-- https://github.com/Naereen/gym-nes-mario-bros
-- MIT License https://lbesson.mit-license.org/

-- global variables
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 224
screen = {} -- screen pixels [x,y] = p
pipe_out = nil -- for sending data(output e.g. screen pixels, reward) back to client
pipe_in = nil -- for getting data(input e.g. controller status change) from client
flag_reset = false -- indicates whether a reset is happening

SEP = string.format('%c', 0xFF) -- as separator in communication protocol
IN_SEP = '|'

-- http://www.fceux.com/web/help/fceux.html?LuaScripting.html
COMMAND_TABLE = {
  A = "A",
  B = "B",
  U = "up",
  L = "left",
  D = "down",
  R = "right",
  S = "start",
  E = "select"
}

-- exported common functions start with nes_ prefix
-- called before each episode
function nes_load_state()
  -- load state so we don't have to instruct to skip title screen
  state = savestate.object(10)
  savestate.load(state)
end

function nes_reset()
  flag_reset = true
  nes_load_state()
end

function nes_get_reset_flag()
  return flag_reset
end

function nes_clear_reset_flag()
  flag_reset = false
end

-- read the pipe number from a file
function emulatornumber()
  local f = io.open("/tmp/nesgym-pipe-number.txt", "r")
  local number = 0
  if f then
    number = tonumber(f:read())
  end
  f:close()
  return number
end

-- called once when emulator starts
function nes_init()
  -- FIXME reduce from "maximum" to "normal" if needed
  emu.speedmode("maximum")
  -- emu.speedmode("normal")

  for x = 0, SCREEN_WIDTH - 1 do
    screen[x] = {}
    for y = 0, SCREEN_HEIGHT - 1 do
      screen[x][y] = -1
    end
  end

  -- pipe_prefix = '/tmp/nesgym-pipe'
  pipe_prefix = string.format("/tmp/nesgym-pipe-%i", emulatornumber())
  -- from emulator to client
  pipe_out, _, _ = io.open(pipe_prefix .. "-in", "w")
  -- from client to emulator
  pipe_in, _, _ = io.open(pipe_prefix .. "-out", "r")

  write_to_pipe("ready" .. SEP .. emu.framecount())
end

-- update_screen - get current screen pixels and store them (256 x 224)
-- Palette is a number from 0 to 127 that represents an RGB color (conversion table in python file)
function nes_update_screen()
  local r, g, b, p
  local framecount = emu.framecount()
  -- NES only has y values in the range 8 to 231, so we need to offset y values by 8
  local offset_y = 8

  write_to_pipe_partial("screen" .. SEP .. framecount .. SEP)
  for y = 0, SCREEN_HEIGHT - 1 do
    local screen_string = ""
    for x = 0, SCREEN_WIDTH - 1 do
      r, g, b, p = emu.getscreenpixel(x, y + offset_y, false)
      -- offset p by 20 so the content can never be '\n'
      screen_string = screen_string .. string.format("%c", p+20)
    end
    write_to_pipe_partial(screen_string)
  end
  write_to_pipe_end()
end

function nes_send_data(data)
  -- print("Sending this data:", data, "...")  -- DEBUG
  write_to_pipe("data" .. SEP .. emu.framecount() .. SEP .. data)
end

function nes_process_command()
  if not pipe_in then
    return false
  end

  local line = pipe_in:read()
  if line ~= nil then
    handle_command(line)
    return true
  end

  return false
end

function nes_ask_for_command()
  write_to_pipe("wait_for_command" .. SEP .. emu.framecount())
end

--- private functions
-- handle one command
function handle_command(line)
  local body = split(line, IN_SEP)
  local command = body[1]
  if command == 'reset' then
    nes_reset()
  elseif command == 'loadstate' then
    nes_load_state()
  elseif command == 'joypad' then
    -- joypad command
    local buttons = body[2]
    local joypad_command = {}
    for i = 1, #buttons do
      local btn = buttons:sub(i,i)
      local button = COMMAND_TABLE[buttons:sub(i,i)]
      joypad_command[button] = true
      gui.text(1, 25, button)
    end
    joypad.set(1, joypad_command)
  end
end

-- write_to_pipe - Write data to pipe
function write_to_pipe(data)
  if data and pipe_out then
    pipe_out:write(data .. SEP .. "\n")
    pipe_out:flush()
  end
end

function write_to_pipe_partial(data)
  if data and pipe_out then
    pipe_out:write(data)
  end
end

function write_to_pipe_end()
  if pipe_out then
    pipe_out:write(SEP .. "\n")
    pipe_out:flush()
  end
end

-- split - Splits a string with a specific delimiter
function split(self, delimiter)
    local results = {}
    local start = 1
    local split_start, split_end  = string.find(self, delimiter, start)
    while split_start do
        table.insert(results, string.sub(self, start, split_start - 1))
        start = split_end + 1
        split_start, split_end = string.find(self, delimiter, start)
    end
    table.insert(results, string.sub(self, start))
    return results
end
