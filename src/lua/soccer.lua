-- #!/usr/bin/env lua
-- By Lilian Besson (Naereen)
-- https://github.com/Naereen/gym-nes-mario-bros
-- MIT License https://lbesson.mit-license.org/

require 'nes_interface'

function get_score()
  local p1score = memory.readbyte(0x052e)
  local p2score = memory.readbyte(0x052F)
  return p1score
end

nes_init()

local score = 0

while true do
  -- update screen every screen_update_interval frames
  local frame_skip = 4

  if emu.framecount() % frame_skip == 0 then
    nes_ask_for_command()
    local has_command = nes_process_command()
    if has_command then
      emu.frameadvance()
      local reward = 0
      if nes_get_reset_flag() then
        nes_clear_reset_flag()
        score = 0
        reward = 0
      else
        local new_score = get_score()
        reward = new_score - score
        score = new_score
      end
      nes_send_data(string.format("%02x%02x", reward, score))
      nes_update_screen()
    else
      print('pipe closed')
      break
    end
  else
    -- skip frames
    emu.frameadvance()
  end
end
