run:	run_mario
run_mario:	cleanpipes
	-pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym-pipe-number.txt
	-python3 run-mario.py
	-pkill -9 fceux

# Experiment to launch a second emulator manually
run_mario1:
	# -pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym-pipe-number.txt
	-python3 run-mario.py
	# -pkill -9 fceux

cleanpipes:
	rm -vf /tmp/nesgym-pipe*

# DEBUG bring back first name for this
old_run_soccer:
	-pkill -9 fceux
	clear
	-python3 run-soccer.py
	-pkill -9 fceux
