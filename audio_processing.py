import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import scipy.signal as sps
from scipy.io import wavfile
# import tensorflow_io as tfio


# @tf.function
# def load_wav_16k_mono(filename):
# 	""" Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
# 	file_contents = tf.io.read_file(filename)
# 	wav, sample_rate = tf.audio.decode_wav(
# 			file_contents,
# 			desired_channels=1)
# 	wav = tf.squeeze(wav, axis=-1)
# 	sample_rate = tf.cast(sample_rate, dtype=tf.int64)
# 	wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
# 	return wav

def take_first(ds):
	vals = [(audio, labels) for audio, labels in ds.take(1)][0]  
	ex_audio = vals[0]
	ex_labels = vals[1]
	return ex_audio, ex_labels

def get_spectrogram(waveform):
	# Convert the waveform to a spectrogram via a STFT (Short-Time Fourier Transform)
	spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
	# Obtain the magnitude of the STFT.
	spectrogram = tf.abs(spectrogram)
	# Add a `channels` dimension, so that the spectrogram can be used
	# as image-like input data with convolution layers (which expect
	# shape (`batch_size`, `height`, `width`, `channels`).
	spectrogram = spectrogram[..., tf.newaxis]
	return spectrogram

def plot_spectrogram(ds, label_names):
	example_audio, example_labels = take_first(ds)
	for i in range(3):
		label = label_names[example_labels[i]]
		waveform = example_audio[i]
		spectrogram = get_spectrogram(waveform)
	if len(spectrogram.shape) > 2:
		assert len(spectrogram.shape) == 3
		spectrogram = np.squeeze(spectrogram, axis=-1)
	# Convert the frequencies to log scale and transpose, so that the time is
	# represented on the x-axis (columns).
	# Add an epsilon to avoid taking a log of zero.
	log_spec = np.log(spectrogram.T + np.finfo(float).eps)
	height = log_spec.shape[0]
	width = log_spec.shape[1]
	X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
	Y = range(height)
	fig, axes = plt.subplots(2, figsize=(12, 8))
	timescale = np.arange(waveform.shape[0])
	axes[0].plot(timescale, waveform.numpy())
	axes[0].set_title('Waveform')
	axes[0].set_xlim([0, 16000])
	axes[1].set_title('Spectrogram')
	axes[1].pcolormesh(X, Y, log_spec)
	plt.suptitle(label.title())
	plt.show()

def plot_history(history):
	metrics = history.history
	plt.figure(figsize=(16,6))
	plt.subplot(1,2,1)
	plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
	plt.legend(['loss', 'val_loss'])
	plt.ylim([0, max(plt.ylim())])
	plt.xlabel('Epoch')
	plt.ylabel('Loss [CrossEntropy]')

	plt.subplot(1,2,2)
	plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
	plt.legend(['accuracy', 'val_accuracy'])
	plt.ylim([0, 100])
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy [%]')
	plt.show()

def plot_predict_confusion_matrix(model, test_ds, label_names):
	y_pred = model.predict(test_ds)
	y_true = tf.concat(list(test_ds.map(lambda s,lab: lab)), axis=0)
	confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
	plt.figure(figsize=(10, 8))
	sns.heatmap(confusion_mtx,
				xticklabels=label_names,
				yticklabels=label_names,
				annot=True, fmt='g')
	plt.xlabel('Prediction')
	plt.ylabel('Label')
	plt.show()

def plot_predictions(x, model):
	prediction = model(x)
	x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
	plt.bar(x_labels, tf.nn.softmax(prediction[0]))
	plt.title('No')
	plt.show()

def make_spec_ds(ds):
	return ds.map(
		map_func=lambda audio,label: (get_spectrogram(audio), label),
		num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
	audio = tf.squeeze(audio, axis=-1)
	return audio, labels

class ExportModel(tf.Module):
	def __init__(self, model, labels):
		self.model = model
		self.label_names = labels
		# Accept either a string-filename or a batch of waveforms.
		# YOu could add additional signatures for a single wave, or a ragged-batch. 
		self.__call__.get_concrete_function(
			x=tf.TensorSpec(shape=(), dtype=tf.string))
		self.__call__.get_concrete_function(
			x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
		
	def get_spectrogram(self, waveform):
		# Convert the waveform to a spectrogram via a STFT (Short-Time Fourier Transform)
		spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
		# Obtain the magnitude of the STFT.
		spectrogram = tf.abs(spectrogram)
		# Add a `channels` dimension, so that the spectrogram can be used
		# as image-like input data with convolution layers (which expect
		# shape (`batch_size`, `height`, `width`, `channels`).
		spectrogram = spectrogram[..., tf.newaxis]
		return spectrogram

	@tf.function
	def __call__(self, x):
	# If they pass a string, load the file and decode it. 
		if x.dtype == tf.string:
			x = tf.io.read_file(x)
			x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
			x = tf.squeeze(x, axis=-1)
			x = x[tf.newaxis, :]
		x = self.get_spectrogram(x)  
		result = self.model(x, training=False)
		class_ids = tf.argmax(result, axis=-1)
		class_names = tf.gather(self.label_names, class_ids)
		return {'predictions':result,
				'class_ids': class_ids,
				'class_names': class_names}

def train_val_label_data(data_dir):
	train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
		directory=data_dir,
		batch_size=64,
		validation_split=0.2,
		seed=0,
		output_sequence_length=16000,
		subset='both')
	labels = np.array(train_ds.class_names)
	return train_ds, val_ds, labels


def build_audio_recognition_model(input_shape, train_ds, labels):
	num_labels = len(labels)
	# Instantiate the `tf.keras.layers.Normalization` layer.
	norm_layer = tf.keras.layers.Normalization()
	# Fit the state of the layer to the spectrograms
	norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))
	model = tf.keras.models.Sequential([
		tf.keras.layers.Input(shape=input_shape),
		# Downsample the input.
		tf.keras.layers.Resizing(32, 32),
		# Normalize.
		norm_layer,
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.Conv2D(64, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.25),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(num_labels),
	])
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'],
	)
	return model

def build_model_lite(X, y, epochs=40, batch_size=32):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Input(shape=(112,), name="input_embedding"))
	model.add(tf.keras.layers.Dense(12, activation="relu"))
	model.add(tf.keras.layers.Dense(8, activation="relu"))
	model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
	model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
	return model

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

def extract_features(audio_file_path, window_size=1024, num_bins=16, target_sample_rate=None):
	sample_rate, audio_data = wavfile.read(audio_file_path)
	if target_sample_rate != None:
		sample_rate = target_sample_rate
	resampled_audio = sps.resample(audio_data, sample_rate)
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

def load_wav_16k_mono_dynamic(fname):
    sample_rate, data = wavfile.read(fname)
    rate_out=16000
    n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, n_samples)
    return wav

def load_wav_16k_mono(fname, rate_out=16000):
    sample_rate, data = wavfile.read(fname)
    # n_samples = round(len(data) * rate_out / sample_rate)
    n_samples = rate_out
    wav = sps.resample(data, n_samples)
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
        xfeatures = load_wav_16k_mono(wav_file)
        feature_arr.append(xfeatures)
        labels.append(red)
    green_files = [os.path.join(green_dir, fname) for fname in os.listdir(green_dir)]
    for wav_file in green_files:
        xfeatures = load_wav_16k_mono(wav_file)
        feature_arr.append(xfeatures)
        labels.append(green)
    blue_files = [os.path.join(blue_dir, fname) for fname in os.listdir(blue_dir)]
    for wav_file in blue_files:
        xfeatures = load_wav_16k_mono(wav_file)
        feature_arr.append(xfeatures)
        labels.append(blue)
    return np.array(feature_arr), np.array(labels)