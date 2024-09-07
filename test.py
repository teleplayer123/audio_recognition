from lib.red_model import score
import numpy as np
import os
from audio_processing import extract_features

red0 = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "red", "red0.wav")
arr = extract_features(red0)
s = score(arr)
print("score: {}".format(s))