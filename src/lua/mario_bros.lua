-- Lua script to load the score for "Mario Bros." game with Fceux
-- Reference for the ROM adress is http://datacrystal.romhacking.net/wiki/Mario_Bros.:RAM_map

require 'nes_interface'

-- FIXME change when tackling the 2-player mode!
function get_score()
  -- 0x0095-0x0097 	Player 1 Score
  -- 0x0099-0x009B 	Player 2 Score
  -- local p1score = memory.readbyte(0x0095)
  -- local p2score = memory.readbyte(0x0099)
  -- memory.readbyterange function?
  -- Ref: http://tasvideos.org/Bizhawk/LuaFunctions.html
  local p1score = memory.readbyterange(0x0095, 2)
  local p2score = memory.readbyterange(0x0099, 2)
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
