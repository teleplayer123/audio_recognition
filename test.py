import tensorflow as tf
import numpy as np
import os
from utils import load_wav_mono, extract_embedding


rgb_save_path = os.path.join(os.getcwd(), "models", "rgb_model", "rgb_model.keras")
rmodel = tf.keras.models.load_model(rgb_save_path)
print(rmodel.summary())

serving_save_dir = os.path.join(os.getcwd(), "models", "serving_models")
serving_save_path = os.path.join(serving_save_dir, "rgb_serving_model.keras")
smodel = tf.keras.models.load_model(serving_save_path)


test_file = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "blue", "blue17.wav")
test_data = load_wav_mono(test_file)
test_data = extract_embedding(test_data, "blue")
class_dict = {0: "red", 1: "green", 2: "blue"}

res = rmodel.predict(test_data[0])
guess = class_dict[np.argmax(np.asarray(res))]
print(guess)