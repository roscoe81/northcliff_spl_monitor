import ST7735
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import RobotoMedium as UserFont
import sounddevice
import numpy
import math
import sys

print("""northcliff_spl_monitor.py Version 0.7 - Monitor and display approximate Sound Pressure Levels

Disclaimer: Not to be used for accurate sound level measurements.
Only measures a limited bandwidth, has a limited method of frequency compensation and requires calibration.

Press Ctrl+C to exit!

""")


class Noise():
    # From https://github.com/pimoroni/enviroplus-python/blob/master/library/enviroplus/noise.py
    # with a change from mean to RMS amplitude measurements in the get_amplitudes_at_frequency_ranges method and
    # the addition of device = "dmic_sv" in the _record method
    
    def __init__(self,
                 sample_rate=16000,
                 duration=0.5):
        """Noise measurement.

        :param sample_rate: Sample rate in Hz
        :param duraton: Duration, in seconds, of noise sample capture

        """

        self.duration = duration
        self.sample_rate = sample_rate

    def get_amplitudes_at_frequency_ranges(self, ranges):
        """Return the RMS amplitude of frequencies in the given ranges.

        :param ranges: List of ranges including a start and end range

        """
        recording = self._record()
        magnitude = numpy.square(numpy.abs(numpy.fft.rfft(recording[:, 0], n=self.sample_rate)))
        result = []
        for r in ranges:
            start, end = r
            result.append(numpy.sqrt(numpy.mean(magnitude[start:end])))
        return result


    def _record(self):
        return sounddevice.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            device = "dmic_sv",
            blocking=True,
            channels=1,
            dtype='float64'
        )

noise = Noise()

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
img = Image.new('RGB', (WIDTH, HEIGHT), color=back_colour)
draw = ImageDraw.Draw(img)

    
while True:
    amps = noise.get_amplitudes_at_frequency_ranges([(30, 100), (100, 1000), (1000, 8000)])
    if display_type <= 1:
        amps[0] *= 0.0562 # Adjust lowest frequencies RMS level by -25dB to appromimate A compensation curve
        amps[1] *= 0.316 # Adjust upper low frequencies RMS level by -10db to appromimate A compensation curve
        weighted_level = sum(amps)/3 # Take mean of adjusted RMS levels
        ref_level = 0.002 # Sets quiet level reference baseline for dB measurements. Can be used for sound level baseline calibration
        calib = 3.1 # Provides an offset for the weighted total RMS level that removes internal microphone noise.
        # calib can be used for sound level gain compensation, in combination with setting the mocrophone gain level via alsamixer
        #print (amps, weighted_level)
        spl_ratio = (weighted_level-calib)/ref_level
        if spl_ratio > 0:
            spl = 20*math.log10(spl_ratio)
            if spl<=40:
                message_colour = (0, 255, 0)
            elif 40<spl<=85:
                message_colour=(255, 255, 0)
            else:
                message_colour = (255, 0, 0)
            #print(spl)
            if display_type == 0:
                img = Image.new('RGB', (WIDTH, HEIGHT), color=back_colour)
                draw = ImageDraw.Draw(img)
                draw.text((13,0), "Sound Level", font=mediumfont, fill=(255, 255, 255))
                draw.text((5, 32), f"{spl:.1f} dB(A)", font=largefont, fill=message_colour)
                disp.display(img)
            else:
                draw.rectangle((0, 0, WIDTH, 14), back_colour)
                img2 = img.copy()
                draw.rectangle((0, 0, WIDTH, HEIGHT), back_colour)
                img.paste(img2, (-6, 0))
                draw.text((30,0), "Sound Level", font=smallfont, fill=(255, 255, 255))
                draw.line((WIDTH, HEIGHT, WIDTH, HEIGHT - (spl/2)), fill=message_colour, width=10)
                disp.display(img)
        else:
            print("Weighted RMS Level is less than the calibration offset")
            print(f"Weighted RMS Level: {weighted_level:.2f}")
            print(f"Calibration Offset: {calib:.2f}")
            print(f"Lowest Frequencies RMS Level: {amps[0]:.2f}")
            print(f"Upper Low Frequencies RMS Level: {amps[1]:.2f}")
            print(f"Highest Frequencies RMS Level: {amps[2]:.2f}")
    else:
        amps[0] *= 0.0562 # Adjust lowest frequencies RMS level by -25dB to appromimate A compensation curve
        amps[1] *= 0.316 # Adjust upper low frequencies RMS level by -10db to appromimate A compensation curve
        amps = [n * 2 for n in amps] # Scale for display
        #print (amps)
        draw.rectangle((0, 0, WIDTH, 17), back_colour)
        img2 = img.copy()
        draw.rectangle((0, 0, WIDTH, HEIGHT), back_colour)
        img.paste(img2, (-16, 0))
        draw.text((15,0), "Frequency Levels", font=smallfont, fill=(255, 255, 255))
        draw.line((WIDTH -15, HEIGHT, WIDTH -15, HEIGHT - amps[0]), fill=(0, 0, 255), width=5)
        draw.line((WIDTH - 10, HEIGHT, WIDTH - 10, HEIGHT - amps[1]), fill=(0, 255, 0), width=5)
        draw.line((WIDTH -5, HEIGHT, WIDTH -5, HEIGHT - amps[2]), fill=(255, 0, 0), width=5)
        disp.display(img)
        
        
    
                
        