-- #!/usr/bin/env lua
-- By Lilian Besson (Naereen)
-- https://github.com/Naereen/gym-nes-mario-bros
-- MIT License https://lbesson.mit-license.org/

-- Set the speed of the emulator
emu.speedmode("normal");

-- Declare and set variables or functions if needed

while true do
    -- Execute instructions for FCEUX
    gui.text(1, 10, "Lua scripting the game!");
    local d3 = memory.readbyteunsigned(0x0095);
    local d2 = memory.readbyteunsigned(0x0096);
    local d1 = memory.readbyteunsigned(0x0097);
    local score = 10000 * (10 * ((d3 - (d3 % 16)) / 16) + (d3 % 16)) + 100 * (10 * ((d2 - (d2 % 16)) / 16) + (d2 % 16)) + (10 * ((d1 - (d1 % 16)) / 16) + (d1 % 16));
    gui.text(1, 40, "Score:");
    gui.text(31, 40, score);
    -- now compute the real score
    -- This essentially tells FCEUX to keep running
    emu.frameadvance();
end;