require 'nes_interface'

frame_skip = 4 -- update screen every screen_update_interval frames

function get_reward()
  p1score = memory.readbyte(0x052e)
  p2score = memory.readbyte(0x052F)
  gui.text(5,10, p1score)
  gui.text(15,10, p2score)
end

nes_init()

while true do
  if emu.framecount() % frame_skip == 0 then
    nes_ask_for_command()
    local has_command = nes_process_command()
    if has_command then
      emu.frameadvance()
      get_reward()
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
