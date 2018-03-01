run:	run_mario
run_mario:	cleanpipes
	-pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym_pipenumber.txt
	-python3 run-mario.py
	-pkill -9 fceux

run_mario1:
	# -pkill -9 fceux
	clear
	# echo 0 > /tmp/nesgym_pipenumber.txt
	-python3 run-mario.py
	# -pkill -9 fceux

cleanpipes:
	rm -vf /tmp/nesgym_pipe*

run_soccer:
	-pkill -9 fceux
	clear
	-python3 run-soccer.py
	-pkill -9 fceux
