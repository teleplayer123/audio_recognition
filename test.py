from utils import play_wavfile
import os
from scipy.io import wavfile
import scipy.signal as sps


def load_wav_16k_mono(fname):
    sample_rate, data = wavfile.read(fname)
    rate_out=16000
    n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, n_samples)
    return wav

fname = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "red", "red5.wav")
wav = load_wav_16k_mono(fname)
play_wavfile(fname)