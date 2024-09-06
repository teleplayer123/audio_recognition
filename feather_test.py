import time
import array
import math
import board
import audiobusio
import neopixel
from ulab import numpy as np
import ulab
import gc


# Helper functions to remove DC bias and compute RMS
def mean(values):
    return sum(values) / len(values)

def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(float(sample - minbuf) * (sample - minbuf) for sample in values)
    return math.sqrt(samples_sum / len(values))

def read_audio_data(mic):
    n_samples = 8192
    data = array.array("H", [0] * n_samples)
    mic.record(data, n_samples)
    return data

def get_spectrogram(data):
    spect = ulab.utils.spectrogram(data)
    spect[0] = 0
    return spect

def avg_spectrogram(data, n_bins):
    n_chunks = len(data) // n_bins
    res = [np.mean(data[i*n_bins:(i+1)*n_bins]) for i in range(n_chunks)]
    return res

def convert_spectrogram(data):
    n_bins = 32
    fft_size = 64
    res = []
    for i in range(0, len(data), fft_size):
        start = i * fft_size
        end = start + fft_size
        chunk = data[start:end]
        if len(chunk) != fft_size:
            continue
        spec = get_spectrogram(chunk[:len(chunk)//2])
        mspec = avg_spectrogram(spec, n_bins)
        res.extend(mspec)
    res = normalized_rms(np.array(res))
    return res

def set_color(led, color):
    if color.lower() == "red":
        led[0] = (255, 0, 0)
    elif color.lower() == "green":
        led[0] = (0, 255, 0)
    elif color.lower() == "blue":
        led[0] = (0, 0, 255)
    else:
        led[0] = (0, 0, 0)

#setup led
led = neopixel.NeoPixel(board.NEOPIXEL, 1)
led.brightness = 0.3

#setup mic
mic = audiobusio.PDMIn(board.TX, board.D12, sample_rate=16000, bit_depth=16)
i = 0
while True:
    data = read_audio_data(mic)
    print(i)
    if sum(data) > 1:
        break
    i+=1
print(data)
