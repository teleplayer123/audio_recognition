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
    data = np.zeros(n_samples*2)
    for i in range(n_samples):
        data[i] = 100*((a.read_u16() * 3.3 / 65536) - 1.65)
    return data

def downsample_waveform(waveform, n_bins):
    waveforms = np.zeros(n_bins)
    n_points = len(waveform) // n_bins
    for i in range(n_bins):
        start = i * n_points
        end = start + n_points
        waveforms[i] = np.mean(waveform[start:end])
    return waveforms

def convert_spectrogram(data):
    orig_len = len(data)
    fft_size = 1024
    n_bins = 16
    res = []
    for i in range(0, orig_len, fft_size):
        chunk = data[i:i+fft_size]
        spect = ulab.utils.spectrogram(chunk)
        spect[0] = 0
        mres = downsample_waveform(spect, n_bins)
        del spect
        gc.collect()
        res.extend(mres)
        del mres
        gc.collect()
    res = np.array(res)
    max_arg = np.max(res)
    min_arg = np.min(res)
    res = (res - min_arg) / (max_arg - min_arg)
    del max_arg
    del min_arg
    gc.collect()
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

while True:
    data = read_audio_data(a0)
    if sum(data) > 1:
        spectrogram = convert_spectrogram(data)
        print("Spectrogram Length: {}".format(len(spectrogram)))
        res = svm_red.score(spectrogram)
        print("Red Score: {}".format(res))
        res = svm_green.score(spectrogram)
        print("Green Score: {}".format(res))
        res = svm_blue.score(spectrogram)
        print("Blue Score: {}".format(res))
        del res
    del data
    gc.collect()

