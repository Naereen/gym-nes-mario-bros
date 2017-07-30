require 'nes_interface'

screen_update_interval = 5 -- update screen every screen_update_interval frames

function get_reward()
  p1score = memory.readbyte(0x052e)
  p2score = memory.readbyte(0x052F)
  gui.text(5,10, p1score)
  gui.text(15,10, p2score)
end

nes_init()

while true do
  nes_ask_for_command()
  nes_process_command()

  emu.frameadvance()
  get_reward()
  if emu.framecount() % screen_update_interval == 0 then
    nes_update_screen()
  end
end
