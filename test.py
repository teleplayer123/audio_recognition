import os
import numpy as np
import matplotlib.pyplot as plt
from models.models_lib import blue_model
from models.models_lib import green_model
from models.models_lib import red_model


def convert_psd_spectrogram(x, fft_size=64):
    num_rows = len(x) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    return spectrogram

def show_blue_spectrogram():
    data = []
    f = os.path.join(os.getcwd(), "blue.txt")
    with open(f, "r") as fh:
        for line in fh:
            data.append(float(line))
    data = np.array(data)
    spec = convert_psd_spectrogram(data)
    a = spec
    plt.figure(figsize=(12, 8))
    rows = 3
    cols = 3
    for i in range(9):
        data = a[i]
        plt.subplot(rows, cols, i+1)
        plt.plot(data)
    plt.show()

def show_green_spectrogram():
    data = []
    f = os.path.join(os.getcwd(), "green.txt")
    with open(f, "r") as fh:
        for line in fh:
            data.append(float(line))
    data = np.array(data)
    spec = convert_psd_spectrogram(data)
    a = spec
    plt.figure(figsize=(12, 8))
    rows = 3
    cols = 3
    for i in range(9):
        data = a[i]
        plt.subplot(rows, cols, i+1)
        plt.plot(data)
    plt.show()

def show_red_spectrogram():
    data = []
    f = os.path.join(os.getcwd(), "red.txt")
    with open(f, "r") as fh:
        for line in fh:
            data.append(float(line))
    data = np.array(data)
    spec = convert_psd_spectrogram(data)
    a = spec
    plt.figure(figsize=(12, 8))
    rows = 3
    cols = 3
    for i in range(9):
        data = a[i]
        plt.subplot(rows, cols, i+1)
        plt.plot(data)
    plt.show()

def show_color_spectrograph(data):
    spec = convert_psd_spectrogram(data)
    h = np.shape(spec)[0]
    w = np.shape(spec)[1]
    x = np.linspace(0, np.size(spec), num=w, dtype=int)
    y = range(h)
    plt.pcolormesh(x, y, spec)
    plt.show()

def get_data(filename):
    data = []
    f = os.path.join(os.getcwd(), filename)
    with open(f, "r") as fh:
        for line in fh:
            data.append(float(line))
    data = np.array(data)
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
    n_bins = 8
    fft_size = 64
    res = []
    for i in range(0, len(data), fft_size):
        spec = np.fft.fft(data[i:i+fft_size])
        spec = np.abs(spec)
        spec[0] = 0
        mspec = downsample_waveform(spec, n_bins)
        res.extend(mspec)
    res = np.array(res)
    amax = np.max(res)
    amin = np.min(res)
    nres = (res - amin) / (amax - amin)
    return np.array(nres)

data = get_data("blue.txt")
print(len(data))
data = convert_spectrogram(data)
print("\nScores for Blue Data\n------------------------")
s = blue_model.score(data)
print("Blue Score: {}".format(s))

s = green_model.score(data)
print("Green Score: {}".format(s))

s = red_model.score(data)
print("Red Score: {}".format(s))

data = get_data("green.txt")
data = convert_spectrogram(data)
print("\n\nScores for Green Data\n------------------------")
s = blue_model.score(data)
print("Blue Score: {}".format(s))

s = green_model.score(data)
print("Green Score: {}".format(s))

s = red_model.score(data)
print("Red Score: {}".format(s))

data = get_data("red.txt")
data = convert_spectrogram(data)
print("\n\nScores for Red Data\n------------------------")
s = blue_model.score(data)
print("Blue Score: {}".format(s))

s = green_model.score(data)
print("Green Score: {}".format(s))

s = red_model.score(data)
print("Red Score: {}".format(s))

show_blue_spectrogram()