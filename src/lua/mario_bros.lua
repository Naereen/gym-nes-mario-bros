-- Lua script to load the score for "Mario Bros." game with Fceux
-- Reference for the ROM adress is http://datacrystal.romhacking.net/wiki/Mario_Bros.:RAM_map
--
-- 0x0029 	Current Game Mode (0=1P A, 1=1P B, 2=2P A, 3=2P B)
-- 0x003A 	Game A (0)/B (1) flag
-- 0x0041 	Current displayed level number
-- 0x0048 	Player 1 Lives
-- 0x004C 	Player 2 Lives
-- 0x0070 	POW hits remaining
-- 0x0071 	Screen shake timer
-- 0x0091-0x0093 	High Score
-- 0x0095-0x0097 	Player 1 Score
-- 0x0099-0x009B 	Player 2 Score
-- 0x04B1 	Bonus Timer (seconds)
-- 0x04B2 	Bonus Timer (milliseconds?)
-- 0x04B5 	Bonus Coins collected P1
-- 0x04B6 	Bonus Coins collected P2
--
-- See http://www.fceux.com/web/help/LuaFunctionsList.html for the list of functions in fceux

require 'nes_interface'

-- TODO change when tackling the 2-player mode!
-- 0x0095-0x0097 	Player 1 Score
-- 0x0099-0x009B 	Player 2 Score
function get_score()
  local d3 = memory.readbyteunsigned(0x0095)
  local d2 = memory.readbyteunsigned(0x0096)
  local d1 = memory.readbyteunsigned(0x0097)
  local p1score = 10000 * (10 * ((d3 - (d3 % 16)) / 16) + (d3 % 16)) + 100 * (10 * ((d2 - (d2 % 16)) / 16) + (d2 % 16)) + (10 * ((d1 - (d1 % 16)) / 16) + (d1 % 16))

  -- local d3 = memory.readbyteunsigned(0x0099)
  -- local d2 = memory.readbyteunsigned(0x009A)
  -- local d1 = memory.readbyteunsigned(0x009B)
  -- local p2score = 10000 * (10 * ((d3 - (d3 % 16)) / 16) + (d3 % 16)) + 100 * (10 * ((d2 - (d2 % 16)) / 16) + (d2 % 16)) + (10 * ((d1 - (d1 % 16)) / 16) + (d1 % 16))

  return p1score
end

nes_init()

local score = 0

-- update screen every screen_update_interval frames
local frame_skip = 4
local frame_skip_debug = 40

while true do
  -- Debugging message
  gui.text(1, 10, "Python/Lua DeepQNLearning - By Naereen")

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
        -- local new_score = 0  -- DEBUG
        local new_score = get_score()
        reward = new_score - score
        score = new_score
        -- gui.text(1, 45, "Reward:")
        -- gui.text(31, 45, reward)
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
