import time
import array
import math
import board
import audiobusio
import neopixel
from ulab import numpy as np
import ulab


# Helper functions to remove DC bias and compute RMS
def mean(values):
    return sum(values) / len(values)

def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(float(sample - minbuf) * (sample - minbuf) for sample in values)
    return math.sqrt(samples_sum / len(values))

def read_audio_data(a):
    data = []
    n_samples = 8000
    n_step = 1024
    for i in range(0, n_samples * 2, n_step):
        data.append(100*((a.read_u16() * 3.3 / 65536) - 1.65))
    if len(data) % 2 != 0:
        data = data[1:]
    return np.array(data)

def get_spectrogram(data):
    spect = ulab.utils.spectrogram(data)
    spect[0] = 0
    return spect

def avg_spectrogram(data, n_bins):
    n_chunks = len(data) // n_bins
    res = [np.mean(data[i*n_bins:(i+1)*n_bins]) for i in range(n_chunks)]
    return res

def convert_spectrogram(data):
    n_bins = 16
    fft_size = 1024
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
    res = np.array(res)
    # np.savetxt("/sd/spectrogram1.txt", res)
    res_min = np.min(res)
    res_max = np.max(res)
    res = (res - res_min) / (res_max - res_min)
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

# Set up the microphone
mic = audiobusio.PDMIn(board.TX, board.D12, sample_rate=8000, bit_depth=16)
samples = array.array('H', [0] * 160)

while True:
    mic.record(samples, len(samples))
    magnitude = normalized_rms(samples)
    print((magnitude,))
    time.sleep(0.1)
