# Makefile to send this to Zam
SHELL=/usr/bin/env /bin/bash

# Senders:
send:	send_zamok
send_zamok:
	CP --exclude=.git ./ ${Szam}publis/gym-nes-mario-bros.git/

send_ws3:
	CP ./ ${Sw}gym-nes-mario-bros.git/
