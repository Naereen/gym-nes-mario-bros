emu.speedmode("normal") -- Set the speed of the emulator

-- Declare and set variables or functions if needed

f = 0
while true do
  -- Execute instructions for FCEUX
  emu.frameadvance() -- This essentially tells FCEUX to keep running
  print('frame', f)
  f = f + 1
end
