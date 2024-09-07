import time
import array
import board
import audiobusio
import neopixel
from ulab import numpy as np
import ulab
import gc
import svm_red
import svm_green
import svm_blue



# Helper functions to remove DC bias and compute RMS
def mean(values):
    return sum(values) / len(values)

def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(float(sample - minbuf) * (sample - minbuf) for sample in values)
    return np.sqrt(samples_sum / len(values))

def read_audio_data(mic):
    n_samples = 8192
    data = array.array("H", [0] * n_samples)
    mic.record(data, n_samples)
    return np.array(data)

def get_spectrogram(data):
    spect = ulab.utils.spectrogram(data)
    spect[0] = 0
    return spect

def downsample_waveform(waveform, n_bins):
    waveforms = np.zeros(n_bins)
    waveform = np.array(waveform)
    n_points = len(waveform) // n_bins
    for i in range(n_bins):
        start = i * n_points
        end = start + n_points
        waveforms[i] = np.mean(waveform[start:end])
    return waveforms

def convert_spectrogram(data):
    orig_len = 8192 // 32
    n_bins = 16
    fft_size = 32
    res = []
    for i in range(0, orig_len, fft_size):
        spec = get_spectrogram(data[i*fft_size:i*fft_size+fft_size])
        mspec = downsample_waveform(spec, n_bins)
        res.extend(mspec)
    return np.array(res)

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

while True:
    data = read_audio_data(mic)
    if len(data) >= 8192:
        print(len(data))
        spec = convert_spectrogram(data)
        print(len(spec))
        rscore = svm_red.score(spec)
        print("Red Score: {}".format(rscore))
        gscore = svm_green.score(spec)
        print("Green Score: {}".format(gscore))
        bscore = svm_blue.score(spec)
        print("Blue Score: {}".format(bscore))
        del data
        del spec
        gc.collect()

