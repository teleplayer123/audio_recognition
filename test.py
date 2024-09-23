import tensorflow as tf
import numpy as np
import os
from utils import load_wav_mono, extract_embedding, load_yamnet_model


rgb_save_path = os.path.join(os.getcwd(), "models", "rgb_model", "rgb_model.keras")
rmodel = tf.keras.models.load_model(rgb_save_path)
print(rmodel.summary())

yamnet_model_dir = os.path.join(os.getcwd(), "models", "yamnet")
smodel = load_yamnet_model(yamnet_model_dir)[0]

test_file = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "blue", "blue17.wav")
test_data = load_wav_mono(test_file)
test_data = extract_embedding(smodel, test_data, "blue")
class_dict = {0: "red", 1: "green", 2: "blue"}

res = rmodel.predict(test_data[0])
guess = class_dict[np.argmax(np.asarray(res))]
print(guess)