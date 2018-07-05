run:	mario
mario:	clean
	-pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym-pipe-number.txt
	-python3 run-mario.py
	-pkill -9 fceux

# Experiment to launch a second emulator manually
mario1:
	# -pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym-pipe-number.txt
	-python3 run-mario.py
	# -pkill -9 fceux

clean:	cleanscores cleanpipes

cleanscores:
	rm -vf *h5 *json *csv

cleanpipes:
	rm -vf /tmp/nesgym-pipe*

soccer:	cleanscores
	-pkill -9 fceux
	clear
	-python3 run-soccer.py
	-pkill -9 fceux

atari:	cleanscores
	clear
	-python3 run-atari.py

pong:	cleanscores
	clear
	-python3 run-atari.py pong

breakout:	cleanscores
	clear
	-python3 run-atari.py breakout

pacman:	cleanscores
	clear
	-python3 run-atari.py pacman

invaders:	cleanscores
	clear
	-python3 run-atari.py invaders
