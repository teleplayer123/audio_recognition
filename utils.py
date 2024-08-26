import winsound as ws
import os
import pyaudio
import wave
from scipy.io import wavfile
import scipy.signal as sps
import tensorflow as tf
import numpy as np


def play_wavfile(fname):
    chunk = 1024  
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()
    # output = True means play data stream rather than record
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    data = wf.readframes(chunk)
    # wav file is played by writing data to stream
    while True:
        if data == "" or len(data) < 1:
            break
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()

def play_wav_data(wavdata):
    """probably don't use this function"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True)
    data = wavdata.astype(np.float32).tostring()
    stream.write(data)
    stream.close()
    p.terminate()

def play_wavedir(path):
    file_list = [os.path.join(path, fname) for fname in os.listdir(path)]
    for wav in file_list:
        ws.PlaySound(wav, ws.SND_FILENAME)

def load_wav_16k_mono(fname):
    sample_rate, data = wavfile.read(fname)
    rate_out=16000
    n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, n_samples)
    return wav