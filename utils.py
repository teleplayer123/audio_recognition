import winsound as ws
import os
import pyaudio
import wave
from scipy.io import wavfile
import scipy.signal as sps
import subprocess
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import math
import pandas as pd


#####################################
#       Convert/Save Functions      #
#####################################

def convert_tflite_model2c(tflite_fname):
    """Supported only on linux/wsl"""
    cmd = "xxd -i {} > model_data.cc".format(tflite_fname)
    res = subprocess.check_output(cmd, encoding="utf-8", shell=True)
    return res

def convert_saved_model_to_tflite(saved_model_dir):
    file_path = os.path.join(os.getcwd(), "models", "model.tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(file_path, "wb") as fh:
        fh.write(tflite_model)

def convert_model_to_tflite(model, outdir="models"):
    file_path = os.path.join(os.getcwd(), outdir, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(file_path, "wb") as fh:
        fh.write(tflite_model)

def save_tf_model(model, outdir="models"):
    model_dir = os.path.join(os.getcwd(), outdir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    save_dir = os.path.join(model_dir, "saved_models")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    tf.saved_model.save(model, save_dir)

def convert_tflite_int8(saved_model_dir, input_shape=(124, 129, 1), n_outputs=8, outdir="models"):
    def representative_dataset():
        for _ in range(n_outputs):
            data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2])
            yield [data.astype(np.float32)]

    file_path = os.path.join(os.getcwd(), outdir, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    with open(file_path, "wb") as fh:
        fh.write(tflite_quant_model)

def save_labels(labels, outfile="labels.txt"):
	savedir = os.path.join(os.getcwd(), "models")
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	save_path = os.path.join(savedir, outfile)
	if not isinstance(labels, type(np.array)):
		labels = np.array(labels)
	labels.tofile(save_path, sep="\n")

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
    data_dict = {k: v for k, v in data.items()}
    return data_dict

def predict_template(path, model_name="red_model"):
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
    outfile = os.path.join(os.getcwd(), "lib", "{}.py".format(model_name))
    with open(outfile, "w") as fh:
        fh.write(template)
    return outfile

############################################
#          Audio Signal Processing         #
############################################

def get_noise(n):
    noise = (np.random.rand(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    return noise

def get_psd(audio, fft_size=1024, sr=8000):
    noise = get_noise(fft_size)
    audio_aug = audio + noise
    spec = np.fft.fftshift(np.fft.fft(audio_aug))
    mag = 10*np.log10(np.abs(spec)**2)
    return mag

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

def prepare_waveform_data(model, waveforms, labels):
    def extract_embedding(waveform, label):
        scores, embeddings, spectrogram = model(waveform)
        num_embed = np.shape(embeddings)[0]
        return embeddings, np.repeat(label, num_embed)
    data = []
    data_labels = []
    for waveform, label in zip(waveforms, labels):
        d, l = extract_embedding(waveform, label)
        data.extend(d)
        data_labels.extend(l)
    data = np.array(data)
    data_labels = np.array(data_labels)
    return data, data_labels

############################################
#               ML Functions               #
############################################

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def shape(t):
    sizes = []
    while isinstance(t, list):
        sizes.append(len(t))
        t = t[0]
    return sizes

def relu(t):
    return np.maximum(0, t)

def normalize(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_(x):
    e_x = [math.exp(i - max(x)) for i in x]
    return [i / sum(e_x) for i in e_x]

############################################
#               Model Functions            #
############################################

def build_model_rgb(X, y, epochs=40, batch_size=32):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Input(shape=(112,), name="input_embedding"))
	model.add(tf.keras.layers.Dense(12, activation="relu"))
	model.add(tf.keras.layers.Dense(8, activation="relu"))
	model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
	model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
	return model

def build_rgb_classifier_layer():
    rgb_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                            name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3)
    ], name='rgb_model')
    rgb_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy'])
    return rgb_model

############################################
#                Load Data                 #
############################################

def load_yamnet_model(model_dir):
    model = tf.saved_model.load(model_dir)
    class_path = model.class_map_path().numpy().decode()
    class_names = list(pd.read_csv(class_path)["display_name"])
    return model, class_names

def load_wav_mono(fname, rate_out=16000):
    sample_rate, data = wavfile.read(fname)
    # n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, rate_out)
    return wav

def extract_embedding(model, waveform, label):
    scores, embeddings, spectrogram = model(waveform)
    num_embed = np.shape(embeddings)[0]
    return embeddings, np.repeat(label, num_embed)

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. 
        Note: Make sure tensorflow and tensorflow_io versions are compatible
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def load_data_rgb_multi_class(data_dir):
    labels = []
    feature_arr = []
    red = 0
    green = 1
    blue = 2
    red_dir = os.path.join(data_dir, "red")
    green_dir = os.path.join(data_dir, "green")
    blue_dir = os.path.join(data_dir, "blue")
    red_files = [os.path.join(red_dir, fname) for fname in os.listdir(red_dir)]
    for wav_file in red_files:
        xfeatures = load_wav_mono(wav_file)
        feature_arr.append(xfeatures.tolist())
        labels.append(red)
    green_files = [os.path.join(green_dir, fname) for fname in os.listdir(green_dir)]
    for wav_file in green_files:
        xfeatures = load_wav_mono(wav_file)
        feature_arr.append(xfeatures.tolist())
        labels.append(green)
    blue_files = [os.path.join(blue_dir, fname) for fname in os.listdir(blue_dir)[:10]]
    for wav_file in blue_files:
        xfeatures = load_wav_mono(wav_file)
        feature_arr.append(xfeatures.tolist())
        labels.append(blue)
    return np.array(feature_arr), np.array(labels)

def load_data(data_dir):
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

def convert_psd_spectrogram(x, fft_size=1024):
    num_rows = len(x) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    return spectrogram

def load_data_rgb(data_dir, color="red"):
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
    blue_files = [os.path.join(blue_dir, fname) for fname in os.listdir(blue_dir)]
    for wav_file in blue_files:
        xfeatures = extract_features(wav_file)
        feature_arr.append(xfeatures)
        labels.append(blue)
    return np.array(feature_arr), np.array(labels)

############################################
#                  Misc                    #
############################################

def set_color(x, s=None, foreground=True):
    """format text with int tuple (RR, GG, BB)"""
    r = x >> 16
    g = (x >> 8) & 0b11111111
    b = x & 0b11111111
    if foreground == True:
        if s == None:
            f = "\033[38;2;{};{};{}m".format(r, g, b)
        else:
            f = "\033[38;2;{};{};{}m {}\033[00m".format(r, g, b, s)
    else:
        if s == None:
            f = "\033[48;2;{};{};{}m".format(r, g, b)
        else:
            f = "\033[48;2;{};{};{}m {}\033[00m".format(r, g, b, s)
    return f

HEATMAP = (0xff0000, 0xff0a00, 0xff1400, 0xff1e00,
           0xff2800, 0xff3200, 0xff3c00, 0xff4600,
           0xff5000, 0xff5a00, 0xff6400, 0xff6e00,
           0xff7800, 0xff8200, 0xff8c00, 0xff9600,
           0xffa000, 0xffaa00, 0xffb400, 0xffbe00,
           0xffc800, 0xffd200, 0xffdc00, 0xffe600,
           0xfff000, 0xfffa00, 0xfdff00, 0xd7ff00,
           0xb0ff00, 0x8aff00, 0x65ff00, 0x3eff00,
           0x17ff00, 0x00ff10, 0x00ff36, 0x00ff5c,
           0x00ff83, 0x00ffa8, 0x00ffd0, 0x00fff4,
           0x00a4ff, 0x0094ff, 0x0084ff, 0x0074ff,
           0x0064ff, 0x0054ff, 0x0044ff, 0x0032ff,
           0x0022ff, 0x0012ff, 0x0002ff, 0x0000ff)

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