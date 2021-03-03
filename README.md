# Northcliff SPL Monitor
A Python Script that performs a basic sound pressure level monitor using the Pimoroni Enviro+

This script explores the potential of using the Pimoroni Enviro+ as a sound pressure level monitor. Its functionality is currently limited due to its use of an approximated A-curve compensation over 3 frequency bands and limited calibration. It should therefore not be used when accurate sound pressure level reading are required and should only be used as a base for future development.

Versions 1.0 and later use streaming to overcome the microphone's startup "plop" that was identified in the excellent review [here](https://flipreview.com/review-of-pimoronis-enviro-board-part2-lcd-noise-level-lightproximity/)

The microphone's startup "plop" can be seen [here](https://github.com/roscoe81/northcliff_spl_monitor/blob/main/Mic%20Graphs/mic_startup_no_offset.png) and it plays havoc with the sound readings if the microphone is started for each sampling. A DC offset remained after removing the startup "plop", seen [here](https://github.com/roscoe81/northcliff_spl_monitor/blob/main/Mic%20Graphs/mic_stable_no_offset.png) and removing that DC results in [this](https://github.com/roscoe81/northcliff_spl_monitor/blob/main/Mic%20Graphs/mic_stable_offset.png).


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

Use alsamixer to set adau7002 capture level to 70


# Operation

For a numerical display of the current approximate overall sound level (Display 0):

Run python3 northcliff_spl_monitor.py or python3 northcliff_spl_monitor.py 0

or for a graphical display of current and past approximate overall sound levels (Display 1):

Run python3 northcliff_spl_monitor.py 1

or for a graphical display of current and past approximate sound levels by frequency band (Display 2):

Run python3 northcliff_spl_monitor.py 2

Version 2.1 adds the ability to cycle through the three displays by briefly touching the Enviro+'s light/proximity sensor

Version 2.2 adds a line in Display 1 that shows the maximum sound level that's been recorded since selecting that display, as well as the time and date when it was recorded. The maximum sound level is reset when selecting another display.
