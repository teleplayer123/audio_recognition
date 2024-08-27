from utils import play_wavfile
import os
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps


def load_wav_16k_mono(fname):
    sample_rate, data = wavfile.read(fname)
    rate_out=16000
    n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, n_samples)
    return wav

fname = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "red", "red0.wav")
# wav = load_wav_16k_mono(fname)
play_wavfile(fname)

# audio read
# origin_sample_rate, origin_audio = wavfile.read(fname)
# wavfile.write('red_orig.wav', origin_sample_rate, origin_audio[:,0])

# origin_num_samples, origin_num_channels = origin_audio.shape

# new_samps = int(origin_num_samples * 16000/44100)

# # resampling
# target_audio_scipy = sps.resample(origin_audio[:,0], new_samps).astype(int)
# target_audio_scipy = np.array(target_audio_scipy, np.int16)