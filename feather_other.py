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
        spec = np.fft.fft(data[i*fft_size:i*fft_size+fft_size])
        mspec = downsample_waveform(spec, n_bins)
        res.extend(mspec)
    return np.array(res)

def get_speech_spectrogram(mic):
    n_samples = 8192
    samples = np.zeros(n_samples)
    
    temp_arr = array.array("H", [0] * 1)
    normalize = lambda val: 100*((val * 3.3 / 65536) - 1.65)
    
    for i in range(n_samples):
        mic.record(temp_arr, 1)
        samples[i] = normalize(temp_arr[0])
        temp_arr = array.array("H", [0] * 1)
        gc.collect()

    if len(samples) % 2 != 0:
        samples = samples[1:]

    gc.collect()

    data_array = ulab.numpy.array(samples,dtype=ulab.numpy.float)
    samples = None
    time.sleep(1)
    gc.collect()

    def calculate_spectrogram_segment(segment):
        spectrogram = ulab.utils.spectrogram(segment)
        spectrogram[0] = 0  # set DC component to 0
        return spectrogram

    def average_bins(spectrogram, bin_size):
        num_bins = len(spectrogram) // bin_size
        averaged_bins = [ulab.numpy.mean(spectrogram[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]
        return averaged_bins

    segment_size = 1024
    bin_size = 32
    result_spectrogram = []

    for start in range(0, len(data_array), segment_size):
        segment = data_array[start:start + segment_size]
        
        if len(segment) == segment_size:
            segment_spectrogram = calculate_spectrogram_segment(segment)
            half_spectrogram = segment_spectrogram[:segment_size // 2]               
            averaged_bins = average_bins(half_spectrogram, bin_size)
            result_spectrogram.extend(averaged_bins)

    data_array = None
    gc.collect()
    result = ulab.numpy.array(result_spectrogram,dtype=ulab.numpy.float)
    result_spectrogram = None
    gc.collect()
    min_val = ulab.numpy.min(result)
    max_val = ulab.numpy.max(result)
    gc.collect()
    normalized_result = (result - min_val) / (max_val - min_val)   

    return normalized_result

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
    spec = get_speech_spectrogram(mic)
    if len(spec) >= 128:
        rscore = svm_red.score(spec)
        print("Red Score: {}".format(rscore))
        gscore = svm_green.score(spec)
        print("Green Score: {}".format(gscore))
        bscore = svm_blue.score(spec)
        print("Blue Score: {}".format(bscore))
        del spec
        gc.collect()
