import os
import numpy as np
import tensorflow as tf
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt


data_dir = os.path.join(os.getcwd(), "rgb_wavs", "rgb")
