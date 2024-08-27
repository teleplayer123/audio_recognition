import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import random


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


def extract_features(audio_file_path, noise_file_path, window_size, overlap,add_noise=True, sample_rate=8192, noise_ampl=0.20, num_bins=16):
    # noise_file_path = "/kaggle/input/google-speech-commands/_background_noise_/white_noise.wav"
    target_sample_rate = sample_rate

    # Read white noise audio file
    noise_sample_rate, noise_data = wavfile.read(noise_file_path)
    noise_duration = len(noise_data) / noise_sample_rate
    
    # Generate random start time within the noise file
    random_start_time = random.uniform(0, noise_duration - 1)
    random_end_time = random_start_time + 1  # 1-second segment

    # Extract random 1-second segment of white noise
    start_index = int(random_start_time * noise_sample_rate)
    end_index = int(random_end_time * noise_sample_rate)
    random_noise_segment = noise_data[start_index:end_index]

    # Resample white noise segment to target sample rate
    resampled_noise_segment = resample(random_noise_segment, target_sample_rate)
    if add_noise:
        resampled_noise_segment = noise_ampl * resampled_noise_segment
    else:
        resampled_noise_segment = 0 * resampled_noise_segment

    # Read and resample audio file
    sample_rate, audio_data = wavfile.read(audio_file_path)
    resampled_audio = resample(audio_data, target_sample_rate)

    # Add white noise to the audio
    augmented_audio = resampled_audio + resampled_noise_segment

    step_size = window_size - overlap
    num_windows = (len(augmented_audio) - window_size) // step_size + 1
    fft_results = []

    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        windowed_signal = augmented_audio[start_index:end_index]
        
        fft_result = np.fft.fft(windowed_signal)
        fft_result = fft_result[0:int(fft_result.shape[0] / 2)]
        fft_magnitude = np.abs(fft_result)
        fft_magnitude[0] = 0
        fft_magnitude = downsample_waveform(fft_magnitude, num_bins)
        fft_results.extend(fft_magnitude)

    return np.array(fft_results)
