import ST7735
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import RobotoMedium as UserFont
import sounddevice as sd
import numpy as np
from numpy import pi, log10
import math
import sys
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from scipy.signal import zpk2tf, zpk2sos, freqs, sosfilt
from waveform_analysis.weighting_filters._filter_design import _zpkbilinear
try:
    # Transitional fix for breaking change in LTR559
    from ltr559 import LTR559
    ltr559 = LTR559()
except ImportError:
    import ltr559


print("""northcliff_spl_monitor.py Version 2.9 - Gen Monitor and display approximate Sound Pressure Levels with improved A-Curve weighting. alsamixer Mic at 10% (2.40dB Gain)

Disclaimer: Not to be used for accurate sound level measurements.

Press Ctrl+C to exit

""")



class Noise():
    def __init__(self, spl_ref_level, log_sound_data, debug_recording_capture, disp, WIDTH, HEIGHT, vsmallfont, smallfont, mediumfont,
                 largefont, back_colour, display_type, img, draw, sample_rate=48000, duration=0.25):
        self.sample_counter = 0
        self.previous_sample_count = 0
        self.spl_ref_level = spl_ref_level
        self.log_sound_data = log_sound_data
        self.debug_recording_capture = debug_recording_capture
        self.disp = disp
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.vsmallfont = vsmallfont
        self.smallfont = smallfont
        self.mediumfont = mediumfont
        self.largefont = largefont
        self.back_colour = back_colour
        self.display_type = display_type
        self.display_changed = False
        self.img = img
        self.draw = draw
        self.duration = duration
        self.sample_rate = sample_rate
        self.last_display_change = 0
        self.max_spl = 0
        self.max_spl_datetime = None
        self.recording = []
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize = 12000, device = "dmic_sv", callback=self.process_frames)

        
    def process_frames(self, recording, frames, time, status):
        self.recording = recording
        self.sample_counter += 1
                
    def restart_stream(self):
        sd.abort()
        sd.start()

    def ABC_weighting(self, curve='A'):
        """
        Design of an analog weighting filter with A, B, or C curve.
        Returns zeros, poles, gain of the filter.
        """
        if curve not in 'ABC':
            raise ValueError('Curve type not understood')

        # ANSI S1.4-1983 C weighting
        #    2 poles on the real axis at "20.6 Hz" HPF
        #    2 poles on the real axis at "12.2 kHz" LPF
        #    -3 dB down points at "10^1.5 (or 31.62) Hz"
        #                         "10^3.9 (or 7943) Hz"
        #
        # IEC 61672 specifies "10^1.5 Hz" and "10^3.9 Hz" points and formulas for
        # derivation.  See _derive_coefficients()

        z = [0, 0]
        p = [-2*pi*20.598997057568145,
             -2*pi*20.598997057568145,
             -2*pi*12194.21714799801,
             -2*pi*12194.21714799801]
        k = 1

        if curve == 'A':
            # ANSI S1.4-1983 A weighting =
            #    Same as C weighting +
            #    2 poles on real axis at "107.7 and 737.9 Hz"
            #
            # IEC 61672 specifies cutoff of "10^2.45 Hz" and formulas for
            # derivation.  See _derive_coefficients()

            p.append(-2*pi*107.65264864304628)
            p.append(-2*pi*737.8622307362899)
            z.append(0)
            z.append(0)

        elif curve == 'B':
            # ANSI S1.4-1983 B weighting
            #    Same as C weighting +
            #    1 pole on real axis at "10^2.2 (or 158.5) Hz"

            p.append(-2*pi*10**2.2)  # exact
            z.append(0)
        b, a = zpk2tf(z, p, k)
        k /= abs(freqs(b, a, [2*pi*1000])[1][0])

        return np.array(z), np.array(p), k



    def A_weighting(self, fs, output='ba'):
        """
        Design of a digital A-weighting filter.
        Designs a digital A-weighting filter for
        sampling frequency `fs`.
        Warning: fs should normally be higher than 20 kHz. For example,
        fs = 48000 yields a class 1-compliant filter.
        Parameters
        ----------
        fs : float
            Sampling frequency
        output : {'ba', 'zpk', 'sos'}, optional
            Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos'). Default is 'ba'.
        Since this uses the bilinear transform, frequency response around fs/2 will
        be inaccurate at lower sampling rates.
        """
        z, p, k = self.ABC_weighting('A')

        # Use the bilinear transformation to get the digital filter.
        z_d, p_d, k_d = _zpkbilinear(z, p, k, fs)

        if output == 'zpk':
            return z_d, p_d, k_d
        elif output in {'ba', 'tf'}:
            return zpk2tf(z_d, p_d, k_d)
        elif output == 'sos':
            return zpk2sos(z_d, p_d, k_d)
        else:
            raise ValueError("'%s' is not a valid output form." % output)

    def A_weight(self, signal, fs):
        sos = self.A_weighting(fs, output='sos')
        return sosfilt(sos, signal)
   
    def get_rms_at_frequency_ranges(self, recording, ranges):
        """Return the RMS levels of frequencies in the given ranges.

        :param ranges: List of ranges including a start and end range

        """
        magnitude = np.square(np.abs(np.fft.rfft(recording[:, 0], n=self.sample_rate)))
        result = []
        for r in ranges:
            start, end = r
            result.append(np.sqrt(np.mean(magnitude[start:end])))
        return result

    def run(self):
        try:
            with self.stream:
                while True:
                    if self.sample_counter != self.previous_sample_count: # Only process new sample
                        self.previous_sample_count = self.sample_counter
                        if self.sample_counter > 10: # Wait for microphone stability
                            recording_offset = np.mean(self.recording)
                            self.recording = self.recording - recording_offset # Remove remaining microphone DC Offset
                            if self.debug_recording_capture: # Option to plot recording sample capture when debugging microphone
                                plt.plot(self.recording)
                                plt.show()
                            weighted_recording = self.A_weight(self.recording, self.sample_rate)
                            weighted_rms = np.sqrt(np.mean(np.square(weighted_recording)))
                            spl_ratio = weighted_rms/self.spl_ref_level
                            if spl_ratio > 0:
                                spl = 20*math.log10(spl_ratio)
                            if spl<=spl_thresholds[0]:
                                message_colour = (0, 255, 0)
                            elif spl_thresholds[0]<spl<=spl_thresholds[1]:
                                message_colour=(255, 255, 0)
                            else:
                                message_colour = (255, 0, 0)
                            if self.display_type == 0:
                                img = self.img
                                draw = self.draw
                                self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                                self.draw.text((13,0), "Noise Level", font=self.mediumfont, fill=message_colour)
                                self.draw.text((5, 32), f"{spl:.1f} dB(A)", font=self.largefont, fill=message_colour)
                                self.disp.display(img)
                            elif self.display_type == 1:
                                # Capture Max sound level once display has been changed for > 2 seconds
                                if spl >= self.max_spl and (time.time() - self.last_display_change) > 2:
                                    self.max_spl = spl
                                    self.max_spl_datetime = datetime.now()
                                    if self.max_spl<=spl_thresholds[0]:
                                        max_spl_colour = (0, 255, 0)
                                    elif spl_thresholds[0]<self.max_spl<=spl_thresholds[1]:
                                        max_spl_colour=(255, 255, 0)
                                    else:
                                        max_spl_colour = (255, 0, 0)
                                self.draw.rectangle((0, 0, self.WIDTH, 14), self.back_colour)
                                self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                                if not self.display_changed:
                                    self.img.paste(img2, (-6, 0))
                                self.draw.line((self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT - (spl-35)), fill=message_colour, width=10) #Scale for display
                                self.draw.rectangle((0, 0, self.WIDTH, 14), self.back_colour)
                                img2 = self.img.copy()
                                self.draw.text((30,0), "Noise Level", font=self.smallfont, fill=message_colour)
                                if self.max_spl != 0:
                                    self.draw.line((0, self.HEIGHT - (self.max_spl-35), self.WIDTH, self.HEIGHT - (self.max_spl-35)), fill=max_spl_colour, width=1) #Display Max Line
                                    date_string = self.max_spl_datetime.strftime("%d %b %y").lstrip('0')
                                    time_string = self.max_spl_datetime.strftime("%H:%M")
                                    if self.max_spl > 85:
                                        text_height = self.HEIGHT - (self.max_spl-37)
                                    else:
                                        text_height = self.HEIGHT - (self.max_spl-20)
                                    self.draw.text((0, text_height), f"Max {self.max_spl:.1f} dB {time_string} {date_string}", font=self.vsmallfont, fill=max_spl_colour)
                                self.disp.display(self.img)
                                self.display_changed = False
                                if self.log_sound_data:
                                    log_data = {"Sample Counter": self.sample_counter, "Mean Amplitude": str(round(recording_offset, 4)) , "Weighted Level": str(weighted_rms)}
                                    with open('<Your log file location and name>', 'a') as f:
                                        f.write (',\n' + json.dumps(log_data))
                            else:
                                amps = self.get_rms_at_frequency_ranges(weighted_recording, [(20, 500), (500, 2000), (2000, 20000)])
                                spl_freq = [0, 0, 0] # Set up spl by frequency list
                                spl_ratio_freq = [n/self.spl_ref_level for n in amps]
                                all_spl_ratio_freq_ok = True
                                for spl_ratio in spl_ratio_freq: # Ensure that ratios are > 0
                                    if spl_ratio <= 0:
                                        all_spl_ratio_freq_ok = False
                                if all_spl_ratio_freq_ok:
                                    for item in range(len(spl_ratio_freq)):
                                        spl_freq[item] = 20*math.log10(spl_ratio_freq[item])
                                    self.draw.rectangle((0, 0, self.WIDTH, 17), self.back_colour)
                                    img2 = self.img.copy()
                                    self.draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), self.back_colour)
                                    if not self.display_changed:
                                        self.img.paste(img2, (-20, 0))
                                    self.draw.text((30,0), "Noise Bands", font=self.smallfont, fill=message_colour)
                                    self.draw.line((self.WIDTH-15, self.HEIGHT, self.WIDTH-15, self.HEIGHT - (spl_freq[0]*1.14-103)), fill=(0, 0, 255), width=5) # Scale for display
                                    self.draw.line((self.WIDTH-10, self.HEIGHT, self.WIDTH-10, self.HEIGHT - (spl_freq[1]*0.844-59)), fill=(0, 255, 0), width=5) # Scale for display
                                    self.draw.line((self.WIDTH-5, self.HEIGHT, self.WIDTH-5, self.HEIGHT - (spl_freq[2]*0.747-45)), fill=(255, 0, 0), width=5) #Scale for display
                                    self.disp.display(self.img)
                                    self.display_changed = False
                    proximity = ltr559.get_proximity()
                    # If the proximity crosses the threshold, toggle the display type
                    if proximity > 1500 and time.time() - self.last_display_change > 1:
                        self.display_type += 1
                        self.display_type %= 3
                        print('Display Type', self.display_type)
                        self.display_changed = True
                        self.max_spl = 0
                        self.max_spl_datetime = None
                        self.last_display_change=time.time()
        except KeyboardInterrupt:
            self.stream.abort()
            print("Keyboard Interrupt")
        
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
vsmallfont = ImageFont.truetype(UserFont, 11)
smallfont = ImageFont.truetype(UserFont, 16)
mediumfont = ImageFont.truetype(UserFont, 24)
largefont = ImageFont.truetype(UserFont, 32)
back_colour = (0, 0, 0)
img = Image.new('RGB', (WIDTH, HEIGHT), color=back_colour)
draw = ImageDraw.Draw(img)
display_type = 0 # Set default display type
if len(sys.argv) > 1:
    display_type = int(sys.argv[1]) # 0 for dB(A) reading, 1 for dB(A) graph, >=2 for RMS(A) level by frequency band
# Set up sound settings
spl_ref_level = 0.000001 # Sets quiet level reference baseline for dB(A) measurements. alsamixer at 10
spl_thresholds = (70, 90)
log_sound_data = False # Set to True to log sound data for debugging
debug_recording_capture = False # Set to True for plotting each recording stream sample
       
if __name__ == '__main__': # This is where to overall code kicks off
    noise = Noise(spl_ref_level, log_sound_data, debug_recording_capture, disp, WIDTH, HEIGHT, vsmallfont, smallfont, mediumfont, largefont, back_colour, display_type, img, draw)
    noise.run()

# Acknowledgements
# A-Weighting from https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py#L29
# get_rms_at_frequency_ranges from https://github.com/pimoroni/enviroplus-python/blob/master/library/enviroplus/noise.py
        
