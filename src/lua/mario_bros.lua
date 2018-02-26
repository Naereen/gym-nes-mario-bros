-- #!/usr/bin/env lua
-- By Lilian Besson (Naereen)
-- https://github.com/Naereen/gym-nes-mario-bros
-- MIT License https://lbesson.mit-license.org/

-- Lua script to load the score for "Mario Bros." game with Fceux
-- Reference for the ROM adress is http://datacrystal.romhacking.net/wiki/Mario_Bros.:RAM_map

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

-- Read RAM game to get level, life, scores etc

function get_level()
  local byte_0 = memory.readbyteunsigned(0x0041)
  local levelnumber = byte_0 % 16
  return levelnumber
end

-- TODO change when tackling the 2-player mode!
function get_life()
  local byte_1 = memory.readbyteunsigned(0x0048)
  local p1life = byte_1 % 256
  -- gui.text(1, 10, "Life:")
  -- gui.text(31, 10, p1life)
  -- local byte_2 = memory.readbyteunsigned(0x004C)
  -- local p2life = byte_2 % 256
  return p1life
end

-- TODO change when tackling the 2-player mode!
function set_life(newlife)
  memory.writebyte(0x0048, newlife)
  -- memory.writebyte(0x004C, newlife)
end


-- TODO change when tackling the 2-player mode!
-- 0x0095-0x0097 	Player 1 Score
-- 0x0099-0x009B 	Player 2 Score
function get_score()
  local byte_1 = memory.readbyteunsigned(0x0095)
  local byte_2 = memory.readbyteunsigned(0x0096)
  local byte_3 = memory.readbyteunsigned(0x0097)
  -- WARNING There is probably a faster way to compute this!
  local p1score = 10000 * (10 * ((byte_1 - (byte_1 % 16)) / 16) + (byte_1 % 16)) + 100 * (10 * ((byte_2 - (byte_2 % 16)) / 16) + (byte_2 % 16)) + (10 * ((byte_3 - (byte_3 % 16)) / 16) + (byte_3 % 16))

  -- local byte_1 = memory.readbyteunsigned(0x0099)
  -- local byte_2 = memory.readbyteunsigned(0x009A)
  -- local byte_3 = memory.readbyteunsigned(0x009B)
  -- WARNING There is probably a faster way to compute this!
  -- local p2score = 10000 * (10 * ((byte_1 - (byte_1 % 16)) / 16) + (byte_1 % 16)) + 100 * (10 * ((byte_2 - (byte_2 % 16)) / 16) + (byte_2 % 16)) + (10 * ((byte_3 - (byte_3 % 16)) / 16) + (byte_3 % 16))

  return p1score
end

nes_init()

score = 0
reward = 0
new_score = 0
-- Default values
default_life = 2
-- default_life = 99  -- Cheating!
default_level = 1
-- Actual values
life = default_life
level = default_level

-- XXX Experimental!
if default_life > 2 then
  print("Magic: bonus life to 2!")
  set_life(default_life)
end


-- update screen every screen_update_interval frames
frame_skip = 4

while true do
  -- Debugging message
  gui.text( 1, 10, "By Naereen")
  gui.text(60, 10, "R:")
  gui.text(70, 10, reward)
  gui.text(85, 10, "L:")
  gui.text(95, 10, life)
  gui.text(125, 10, "S:")
  gui.text(145, 10, score)
  gui.text(170, 10, "#:")
  gui.text(185, 10, level)

  if emu.framecount() % frame_skip == 0 then
    nes_ask_for_command()
    if nes_process_command() then
      emu.frameadvance()
      if nes_get_reset_flag() then
        nes_clear_reset_flag()
        score = 0
        reward = 0
        life = default_life
        level = default_level
      else
        new_score = get_score()
        reward = new_score - score
        score = new_score
        life = get_life()
        level = get_level()
      end
      nes_send_data(string.format("%02x:%06i:%02x:%02x", (reward % 256), score, life, level))
      nes_update_screen()
      -- XXX Experimental infinite life, avoid restarting the game!
      if life < 1 then
        print("Magic: bonus life to 2!")
        set_life(default_life)
        life = default_life
      end
    else
      print('pipe closed')
      break
    end
  else
    -- skip frames
    emu.frameadvance()
  end
end
