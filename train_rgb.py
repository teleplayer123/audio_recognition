import os
import numpy as np
import tensorflow as tf
import scipy.signal as sps
from scipy.io import wavfile


def downsample_waveform(waveform, num_bins):
    waveform = np.array(waveform)
    original_length = len(waveform)
    points_per_bin = original_length // num_bins
    downsampled_waveform = np.zeros(num_bins)
    for i in range(num_bins):
        start_index = i * points_per_bin
        end_index = start_index + points_per_bin
        downsampled_waveform[i] = waveform[start_index:end_index].mean()
    return downsampled_waveform.tolist()

def add_white_noise(audio):
    #generate noise and the scalar multiplier
    noise = tf.random.uniform(shape=tf.shape(audio), minval=-1, maxval=1)
    noise_scalar = tf.random.uniform(shape=[1], minval=0, maxval=0.2)
    # add them to the original audio
    audio_with_noise = audio + (noise * noise_scalar)
    # final clip the values to ensure they are still between -1 and 1
    audio_with_noise = tf.clip_by_value(audio_with_noise, clip_value_min=-1, clip_value_max=1)
    return audio_with_noise

def extract_features(audio_file_path, window_size=1024, num_bins=16, target_sample_rate=8192, overlap=0):
    sample_rate, audio_data = wavfile.read(audio_file_path)
    resampled_audio = sps.resample(audio_data, target_sample_rate)
    augmented_audio = add_white_noise(resampled_audio)
    step_size = window_size - overlap
    num_windows = (len(augmented_audio) - window_size) // step_size + 1
    fft_results = []
    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        windowed_signal = augmented_audio[start_index:end_index]
        fft_result = np.fft.fft(windowed_signal)
        fft_result = fft_result[0:int(fft_result.shape[0] // 2)]
        fft_magnitude = np.abs(fft_result)
        fft_magnitude[0] = 0
        fft_magnitude = downsample_waveform(fft_magnitude, num_bins)
        fft_results.extend(fft_magnitude)
    return np.array(fft_results)


data_dir = os.path.join(os.getcwd(), "rgb_wavs", "rgb")

def load_dataset(data_dir):
    waveforms = []
    labels = []
    for dirname in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, dirname)
        if not dirname in labels:
            labels.append(dirname)
        wav_files = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir)]
        feature_arr = []
        for wav_file in wav_files:
            xfeatures = extract_features(wav_file, window_size=64, num_bins=8)
            feature_arr.append(xfeatures)
        waveforms.append(np.array(feature_arr))
        del feature_arr
    return np.array(waveforms), np.array(labels)

def load_data(data_dir, color="red"):
    labels = []
    feature_arr = []
    red = 0
    green = 0
    blue = 0
    red_dir = os.path.join(data_dir, "red")
    green_dir = os.path.join(data_dir, "green")
    blue_dir = os.path.join(data_dir, "blue")
    if color == "red":
        red = 1
    elif color == "green":
        green = 1
    elif color == "blue":
        blue = 1
    red_files = [os.path.join(red_dir, fname) for fname in os.listdir(red_dir)]
    for wav_file in red_files:
        xfeatures = extract_features(wav_file)
        feature_arr.append(xfeatures)
        labels.append(red)
    green_files = [os.path.join(green_dir, fname) for fname in os.listdir(green_dir)]
    for wav_file in green_files:
        xfeatures = extract_features(wav_file)
        feature_arr.append(xfeatures)
        labels.append(green)
    blue_files = [os.path.join(blue_dir, fname) for fname in os.listdir(blue_dir)[:10]]
    for wav_file in blue_files:
        xfeatures = extract_features(wav_file)
        feature_arr.append(xfeatures)
        labels.append(blue)
    return np.array(feature_arr), np.array(labels)

color = "blue"
audio_data, labels = load_data(data_dir, color=color)

print(np.shape(audio_data))
print(np.shape(labels))

X_train, y_train = audio_data, labels

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

X_norm_train = np.array([normalize(x) for x in X_train])
print(np.shape(X_norm_train))
print(np.shape(y_train))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(128,), name="input_embedding"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_norm_train, y_train, epochs=60, validation_split=0.1, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3))

weights_biases = {}
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    weights_biases[f'w{i}'] = weights
    weights_biases[f'b{i}'] = biases

saved_path = os.path.join(os.getcwd(), "models", f"{color}_weights_biases.npz")
np.savez(saved_path, **weights_biases)
data = np.load(saved_path)

# print(dir(data))
data_dict = {k: v for k, v in data.items()}
# print(data_dict.keys())

def load_weights_biases(path):
    data = np.load(path)
    data_dict = {k: v for k, v in data.items()}
    return data_dict

def predict_template(path, model_name):
    data = load_weights_biases(path)
    template = \
f"""import ulab.numpy as np

def relu(t):
    return np.maximum(0, t)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def score(t):
    z0 = np.dot(t, np.array({data["w0"].tolist()})) + np.array({data["b0"].tolist()})
    a0 = relu(z0)
    z1 = np.dot(a0, np.array({data["w1"].tolist()})) + np.array({data["b1"].tolist()})
    a1 = relu(z1)
    z2 = np.dot(a1, np.array({data["w2"].tolist()})) + np.array({data["b2"].tolist()})
    res = sigmoid(z2)
    return res 
"""
    outfile = os.path.join(os.getcwd(), "models", "models_lib", "{}.py".format(model_name))
    with open(outfile, "w") as fh:
        fh.write(template)
    return outfile

data_path = os.path.join(os.getcwd(), "models", f"{color}_weights_biases.npz")
py_path = predict_template(data_path, model_name=f"{color}_model")
print(py_path)