import ST7735
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import RobotoMedium as UserFont
import sounddevice as sd
import numpy
import math
import sys
import matplotlib.pyplot as plt
import json


print("""northcliff_spl_monitor.py Version 1.0 - Gen - Monitor and display approximate Sound Pressure Levels - Streaming Variant

Disclaimer: Not to be used for accurate sound level measurements.
Only has a limited method of frequency compensation and requires calibration.

Press Ctrl+C to exit

""")


class Noise():
    def __init__(self, spl_ref_level, log_sound_data, debug_recording_capture, disp, WIDTH, HEIGHT, smallfont, mediumfont,
                 largefont, back_colour, display_type, img, draw, sample_rate=48000, duration=0.1):
        self.restart_samples = False
        self.sample_counter = 0
        self.spl_ref_level = spl_ref_level
        self.log_sound_data = log_sound_data
        self.debug_recording_capture = debug_recording_capture
        self.disp = disp
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.smallfont = smallfont
        self.mediumfont = mediumfont
        self.largefont = largefont
        self.back_colour = back_colour
        self.display_type = display_type
        self.img = img
        self.draw = draw
        self.duration = duration
        self.sample_rate = sample_rate
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize = 4800, device = "dmic_sv", callback=self.process_frames)
        self.stream.start()
        
    def process_frames(self, recording, frames, time, status):
        self.sample_counter += 1
        if self.sample_counter > 10: # Wait for microphone stability
            if self.sample_counter <= 400: # Stream for 400 loop cycles before restarting stream
                self.restart_samples = False
                recording_offset = numpy.mean(recording[20:])
                recording = recording - recording_offset # Remove remaining microphone DC Offset
                amps = self.get_rms_at_frequency_ranges(recording[20:], [(30, 100), (100, 500), (500, 20000)]) # Ignore first 20 samples
                amps[0] *= 0.0562 # Adjust lowest frequencies RMS level by -25dB to approximate A compensation curve
                amps[1] *= 0.316 # Adjust upper low frequencies RMS level by -10db to approximate A compensation curve
                # No compensation for mid and high frequencies
                if self.display_type <= 1:
                    weighted_level = sum(amps)/3 # Take mean of adjusted RMS levels
                    spl_ratio = (weighted_level)/self.spl_ref_level
                    if spl_ratio > 0:
                        spl = 20*math.log10(spl_ratio)
                        if spl<=40:
                            message_colour = (0, 255, 0)
                        elif 40<spl<=85:
                            message_colour=(255, 255, 0)
                        else:
                            message_colour = (255, 0, 0)
                        if self.display_type == 0:
                            img = self.img
                            draw = self.draw
                            self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                            self.draw.text((13,0), "Sound Level", font=self.mediumfont, fill=(255, 255, 255))
                            self.draw.text((5, 32), f"{spl:.1f} dB(A)", font=self.largefont, fill=message_colour)
                            self.disp.display(img)
                        else:
                            self.draw.rectangle((0, 0, self.WIDTH, 14), self.back_colour)
                            img2 = self.img.copy()
                            self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                            self.img.paste(img2, (-6, 0))
                            self.draw.text((30,0), "Sound Level", font=self.smallfont, fill=(255, 255, 255))
                            self.draw.line((self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT - (spl-40)), fill=message_colour, width=10) #Scale for display
                            self.disp.display(self.img)
                        if self.log_sound_data:
                            log_data = {"Sample Counter": self.sample_counter, "Mean Amplitude": str(round(recording_offset, 4)) , "Weighted Level": str(round(weighted_level, 4))}
                            with open('<Your log file location and name', 'a') as f:
                                f.write (',\n' + json.dumps(log_data))
                else:
                    spl_freq = [0, 0, 0] # Set up spl by frequency list
                    spl_ratio_freq = [n/self.spl_ref_level for n in amps]
                    all_spl_ratio_freq_ok = True
                    for spl_ratio in spl_ratio_freq: # Ensure that ratios are > 0
                        if spl_ratio <= 0:
                            all_spl_ratio_freq_ok = False
                    if all_spl_ratio_freq_ok:
                        for item in range(len(spl_ratio_freq)):
                            spl_freq[item] = 20*math.log10(spl_ratio_freq[item])
                        #print('SPL Freq', spl_freq)
                        self.draw.rectangle((0, 0, self.WIDTH, 17), self.back_colour)
                        img2 = self.img.copy()
                        self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                        self.img.paste(img2, (-20, 0))
                        self.draw.text((15,0), "Frequency Levels", font=self.smallfont, fill=(255, 255, 255))
                        self.draw.line((self.WIDTH -15, self.HEIGHT, self.WIDTH -15, self.HEIGHT - (spl_freq[0]-40)), fill=(0, 0, 255), width=5) # Scale for display
                        self.draw.line((self.WIDTH - 10, self.HEIGHT, self.WIDTH - 10, self.HEIGHT - (spl_freq[1]-40)), fill=(0, 255, 0), width=5) # Scale for display
                        self.draw.line((self.WIDTH -5, self.HEIGHT, self.WIDTH -5, self.HEIGHT - (spl_freq[2]-40)), fill=(255, 0, 0), width=5) #Scale for display
                        self.disp.display(self.img)
            else: # Restart streaming
                self.sample_counter = 0
                self.restart_samples = True
          
    def get_rms_at_frequency_ranges(self, recording, ranges):
        """Return the RMS levels of frequencies in the given ranges.

        :param ranges: List of ranges including a start and end range

        """
        if self.debug_recording_capture: # Option to plot recording sample capture when debugging microphone
            plt.plot(recording)
            plt.show()
        magnitude = numpy.square(numpy.abs(numpy.fft.rfft(recording[:, 0], n=self.sample_rate)))
        result = []
        for r in ranges:
            start, end = r
            result.append(numpy.sqrt(numpy.mean(magnitude[start:end])))
        return result

# Set up display

disp = ST7735.ST7735(
    port=0,
    cs=ST7735.BG_SPI_CS_FRONT,
    dc=9,
    backlight=12,
    rotation=270)
disp.begin()
WIDTH = disp.width
HEIGHT = disp.height
smallfont = ImageFont.truetype(UserFont, 16)
mediumfont = ImageFont.truetype(UserFont, 24)
largefont = ImageFont.truetype(UserFont, 32)
back_colour = (0, 0, 0)
display_type = 0 # Set default display type
if len(sys.argv) > 1:
    display_type = int(sys.argv[1]) # 0 for dB(A) reading, 1 for dB(A) graph, >=2 for RMS(A) level by frequency band
spl_ref_level = 0.0015 # Sets quiet level reference baseline for dB(A) measurements. Can be used for sound level baseline calibration
log_sound_data = False # Set to True to log sound data for debugging
debug_recording_capture = False # Set to True for plotting each recording stream sample
img = Image.new('RGB', (WIDTH, HEIGHT), color=back_colour)
draw = ImageDraw.Draw(img)
noise = Noise(spl_ref_level, log_sound_data, debug_recording_capture, disp, WIDTH, HEIGHT, smallfont, mediumfont, largefont, back_colour, display_type, img, draw)

try:
    while True:
        if noise.restart_samples: # Restart sounddevice streaming after 400 capture cycles
            noise.stream.stop()
            noise.stream.start()
            noise.restart_samples = False
        else:
            pass
except KeyboardInterrupt:
    noise.stream.stop()
    print("Keyboard Interrupt")
    
        
    
    
                
        
