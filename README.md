# Northcliff SPL Monitor
A Python Script that performs a basic sound pressure level monitor using the Pimoroni Enviro+

This script explores the potential of using the Pimoroni Enviro+ as a sound pressure level monitor. Its functionality is currently limited due to its narrow frequency range, an approximated A-curve compensation over 3 frequency bands and limited calibration. It should therefore not be used when accurate sound pressure level reading are required and should only be used as a base for future development.

# Setup
sudo apt-get update
sudo apt-get-upgrade
curl -sSL https://get.pimoroni.com/enviroplus | bash

sudo python -m pip uninstall sounddevice
sudo python -m pip install sounddevice==0.3.15

Follow instructions at:
https://learn.adafruit.com/adafruit-i2s-mems-microphone-breakout/raspberry-pi-wiring-test
including “Adding Volume Control”

Use the following instead of the documented text for ~/.asoundrc:

1.	#This section makes a reference to your I2S hardware, adjust the card name
2.	#to what is shown in arecord -l after card x: before the name in []
3.	#You may have to adjust channel count also but stick with default first
4.	pcm.dmic_hw {
5.	type hw
6.	card adau7002
7.	channels 2
8.	format S32_LE
9.	}
10.	 
11.	#This is the software volume control, it links to the hardware above and after
12.	#saving the .asoundrc file you can type alsamixer, press F6 to select
13.	#your I2S mic then F4 to set the recording volume and arrow up and down
14.	#to adjust the volume
15.	#After adjusting the volume - go for 50 percent at first, you can do
16.	#something like 
17.	#arecord -D dmic_sv -c2 -r 48000 -f S32_LE -t wav -V mono -v myfile.wav
18.	pcm.dmic_sv {
19.	type softvol
20.	slave.pcm dmic_hw
21.	control {
22.	name "Master Capture Volume"
23.	card adau7002
24.	}
25.	min_dB -3.0
26.	max_dB 30.0
27.	}

Use alsamixer to set adau7002 capture level to 100

