# NESGym

An EXPERIMENTAL [openai-gym](https://gym.openai.com/) wrapper for NES games.

# Installation
- Install the `fceux` NES emulator and make sure `fceux` is in your `$PATH`. In Debian/Ubuntu, simple use `sudo apt install fceux`.
- Copy state files from `roms/fcs/*` to your `~/.fceux/fcs/` (faster loading for the beginning of the game).

# Example usage
For instance:
```python
# import nesgym to register environments to gym
import nesgym
env = gym.make('nesgym/NekketsuSoccerPK-v0')
obs = env.reset()

for step in range(10000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    ... # your awesome reinforcement learning algorithm is here
```

# Examples for training dqn
An implementation of dqn is in [`src/dqn`](src/dqn), using [keras](https://keras.io/).

You can train dqn model for Atari with [`run-atari.py`](src/run-atari.py) and for NES with [`run-soccer.py`](src/run-soccer.py) or [`run-mario.py`](src/run-mario.py).

# Integrating new NES games?
You need to write two files:
1. a lua interface file,
2. and an openai gym environment class(python) file.

The lua file needs to get the reward from emulator(typically extracting from a memory location), and the python file defines the game specific environment.

For an example of lua file, see [`src/lua/soccer.lua`](src/lua/soccer.lua); for an example of gym env file, see [`src/nesgym/nekketsu_soccer_env.py`](src/nesgym/nekketsu_soccer_env.py).

> [This website](http://datacrystal.romhacking.net/wiki/Category:NES_games) gives RAM mapping for the most well-known NES games, this is very useful to extract easily the score or lives directly from the NES RAM memory, to use it as a reward for the reinforcement learning loop. See for instance [for Mario Bros.](http://datacrystal.romhacking.net/wiki/Mario_Bros.:RAM_map).

# Gallery
## Training Atari games
![atari](images/atari.png)

## Training NES games
![fc-soccer](images/soccer.png)
