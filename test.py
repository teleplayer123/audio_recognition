import os
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import matplotlib.pyplot as plt

data = []
f = os.path.join(os.getcwd(), "blue.txt")
with open(f, "r") as fh:
    for line in fh:
        data.append(float(line))

data = np.array(data)

def convert_psd_spectrogram(x, fft_size=64):
    num_rows = len(x) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    return spectrogram

spec = convert_psd_spectrogram(data)

