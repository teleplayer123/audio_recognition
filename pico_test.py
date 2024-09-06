from machine import Pin, PWM, ADC, SPI
import math
import time
from ulab import numpy as np
import ulab
import gc
import svm_red
import svm_green
import svm_blue



def read_audio_data(a):
    n_samples = 8192
    data = np.zeros(n_samples)
    for i in range(n_samples):
        data[i] = 100*((a.read_u16() * 3.3 / 65536) - 1.65)
    return data

def downsample_waveform(waveform, n_bins):
    waveforms = np.zeros(n_bins)
    waveform = np.array(waveform)
    n_points = len(waveform) // n_bins
    for i in range(n_bins):
        start = i * n_points
        end = start + n_points
        waveforms[i] = waveform[start:end].mean()
    return waveforms

def convert_spectrogram(data):    
    fft_size = 64
    n_bins = 32
    res = []
    for i in range(0, len(data), fft_size):
        spect = ulab.utils.spectrogram(data[i*fft_size:i*fft_size+fft_size])
        mres = downsample_waveform(spect, n_bins)
        res.extend(mres)
    res = np.array(res)
    return res
        
    
def set_color(color):
    blue = PWM(Pin(0), freq=1000)
    green = PWM(Pin(1), freq=1000)
    red = PWM(Pin(2), freq=1000)
    if color.lower() == "red":
        red.duty_u16(65535)
        green.duty_u16(0)
        blue.duty_u16(0)
    elif color.lower() == "green":
        red.duty_u16(0)
        green.duty_u16(65535)
        blue.duty_u16(0)
    elif color.lower() == "blue":
        red.duty_u16(0)
        green.duty_u16(0)
        blue.duty_u16(65535)
    elif color.lower() == "off":
        red.duty_u16(0)
        green.duty_u16(0)
        blue.duty_u16(0)
    else:
        print("color must be one of: red, green, blue, or off")

def pulse(p, data):
    for i in range(len(data)):
        p.duty_u16(int(math.sin(i / 10 * math.pi) * 5000 + 5000))
        time.sleep_ms(50)


a0 = ADC(Pin(26))

data = read_audio_data(a0)
spectrogram = convert_spectrogram(data)
print(len(spectrogram))
res = svm_red.score(spectrogram)
print("Red Score: {}".format(res))


