import os
import numpy as np
import tensorflow as tf
import scipy.signal as sps
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.linear_model import SGDClassifier
import m2cgen



data_dir = os.path.join(os.getcwd(), "rgb_wavs", "rgb")

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

def extract_features(audio_file_path, window_size=1024, num_bins=16, target_sample_rate=8192):
    sample_rate, audio_data = wavfile.read(audio_file_path)
    resampled_audio = sps.resample(audio_data, target_sample_rate)
    augmented_audio = add_white_noise(resampled_audio)
    step_size = window_size
    num_windows = len(augmented_audio) // step_size
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

# def load_data(data_dir):
#     labels = []
#     feature_arr = []
#     red_dir = os.path.join(data_dir, "red")
#     green_dir = os.path.join(data_dir, "green")
#     blue_dir = os.path.join(data_dir, "blue")
#     red_files = [os.path.join(red_dir, fname) for fname in os.listdir(red_dir)[:10]]
#     for wav_file in red_files:
#         xfeatures = extract_features(wav_file)
#         feature_arr.append(xfeatures)
#         labels.append(1)
#     green_files = [os.path.join(green_dir, fname) for fname in os.listdir(green_dir)[:10]]
#     for wav_file in green_files:
#         xfeatures = extract_features(wav_file)
#         feature_arr.append(xfeatures)
#         labels.append(0)
#     blue_files = [os.path.join(blue_dir, fname) for fname in os.listdir(blue_dir)[:10]]
#     for wav_file in blue_files:
#         xfeatures = extract_features(wav_file)
#         feature_arr.append(xfeatures)
#         labels.append(0)
#     return np.array(feature_arr), np.array(labels)

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
            xfeatures = extract_features(wav_file)
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
    red_files = [os.path.join(red_dir, fname) for fname in os.listdir(red_dir)[:10]]
    for wav_file in red_files:
        xfeatures = extract_features(wav_file)
        feature_arr.append(xfeatures)
        labels.append(red)
    green_files = [os.path.join(green_dir, fname) for fname in os.listdir(green_dir)[:10]]
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


# audio_data, labels = load_data(data_dir)

# print(np.shape(audio_data))
# print(np.shape(labels))

# x, y = audio_data, labels

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(x.shape)
# print(y.shape)

# new_shape = x.shape[1]*x.shape[2]
# x = np.reshape(x, (3, new_shape))
# y = np.ravel(y)
# model = SVC(kernel="rbf")
# model.fit(x, y)
# score = model.score(x, y)
# print(score)

# code = m2cgen.export_to_python(model)

# with open("svc_rbf_red.py", "w") as fh:
#     fh.write(code)

red_audio_data, red_labels = load_data(data_dir, color="red")
green_audio_data, green_labels = load_data(data_dir, color="green")
blue_audio_data, blue_labels = load_data(data_dir, color="blue")

X_train, X_test, y_train, y_test = train_test_split(red_audio_data, red_labels, test_size=0.2, random_state=42)

def normalize(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

X_norm_train = np.array([normalize(x) for x in X_train])
print(np.shape(X_norm_train))
print(np.shape(y_train))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(112,), name="input_embedding"))
model.add(tf.keras.layers.Dense(12, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_norm_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

def save_weights_biases(model):
    weights_biases = {}
    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        weights_biases["w{}".format(i)] = weights
        weights_biases["b{}".format(i)] = biases

    saved_path = os.path.join(os.getcwd(), "models", "weights_biases.npz")
    np.savez(saved_path, **weights_biases)
    return saved_path

def load_weights_biases(path):
    data = np.load(path)
    