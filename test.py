from models.models128_lib.red_model import score
import numpy as np
import os
from audio_processing import extract_features

red0 = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "red", "red0.wav")
arr = extract_features(red0, target_sample_rate=8192)
s = score(arr)[0]
print("score: {}".format(s))