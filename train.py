import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data_rgb_multi_class, prepare_waveform_data, build_rgb_classifier_layer, load_wav_mono, save_weights_biases, load_weights_biases


RGB_DATA_DIR = os.path.join(os.getcwd(), "rgb_wavs", "rgb")


def load_yamnet_model():
    model_dir = os.path.join(os.getcwd(), "models", "yamnet")
    model = tf.saved_model.load(model_dir)
    class_path = model.class_map_path().numpy().decode()
    class_names = list(pd.read_csv(class_path)["display_name"])
    return model, class_names

def test_yamnet_model(model, class_names, test_data):
    scores, embeddings, spectrogram = model(test_data)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.math.argmax(class_scores)
    inferred_class = class_names[top_class]
    print(f'The main sound is: {inferred_class}')
    print(f'The embeddings shape: {embeddings.shape}')
    print(f"Shape of spectrogram: {np.shape(spectrogram)}")



yamnet_model, yamnet_class_names = load_yamnet_model()
waveforms, labels = load_data_rgb_multi_class(RGB_DATA_DIR)

data, data_labels = prepare_waveform_data(yamnet_model, waveforms, labels)

rgb_model = build_rgb_classifier_layer()
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
history = rgb_model.fit(data, data_labels, epochs=40, validation_split=0.1, callbacks=callback)

test_file = os.path.join(os.getcwd(), "rgb_wavs", "rgb", "blue", "blue14.wav")
test_data = load_wav_mono(test_file)
class_dict = {0: "red", 1: "green", 2: "blue"}

score, embed, spec = yamnet_model(test_data)
res = rgb_model(embed).numpy()
inferred_res = class_dict[res.mean(axis=0).argmax()]
print(f"Inferred Result: {inferred_res}")

#save model weights
weight_bias_path = os.path.join(os.getcwd(), "models", "rgb_yammnet_weights_biases.npz")
fname = "rgb_yammnet_weights_biases"
save_weights_biases(rgb_model, fname)

def predict_template(path, model_name):
    data = load_weights_biases(path)
    template = \
f"""import ulab.numpy as np

def relu(t):
    return np.maximum(0, t)

def score(t):
    z0 = np.dot(t, np.array({data["w0"].tolist()})) + np.array({data["b0"].tolist()})
    a0 = relu(z0)
    res = np.dot(a0, np.array({data["w1"].tolist()})) + np.array({data["b1"].tolist()})
    return res 
"""
    outfile = os.path.join(os.getcwd(), "models", "{}.py".format(model_name))
    with open(outfile, "w") as fh:
        fh.write(template)
    return outfile

predict_template(weight_bias_path, "rgb_yamnet")