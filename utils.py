import winsound as ws
import os
import pyaudio
import wave
from scipy.io import wavfile
import scipy.signal as sps
import tensorflow as tf
import numpy as np


#####################################
#       Audio Play/Processing       #
#####################################

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

def load_wav_16k_mono(fname):
    sample_rate, data = wavfile.read(fname)
    rate_out=16000
    n_samples = round(len(data) * rate_out / sample_rate)
    wav = sps.resample(data, n_samples)
    return wav

def add_white_noise(audio):
    #generate noise and the scalar multiplier
    noise = tf.random.uniform(shape=tf.shape(audio), minval=-1, maxval=1)
    noise_scalar = tf.random.uniform(shape=[1], minval=0, maxval=0.2)
    # add them to the original audio
    audio_with_noise = audio + (noise * noise_scalar)
    # final clip the values to ensure they are still between -1 and 1
    audio_with_noise = tf.clip_by_value(audio_with_noise, clip_value_min=-1, clip_value_max=1)
    return audio_with_noise


#####################################
#       Convert/Save Functions      #
#####################################

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