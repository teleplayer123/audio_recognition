import time
import board
import analogio
import neopixel
import ulab.numpy as np
import ulab
import gc
import red_model
import green_model
import blue_model


gc.collect()
print("Free Memory: {}".format(gc.mem_free()))

def read_audio_data(mic):
    n_samples = 8192
    data = np.empty((n_samples))
    print("speak...")
    time.sleep(1)
    for i in range(n_samples):
        data[i] = (100*((mic.value * 3.3 / 65536) - 1.65))
        gc.collect()
    gc.collect()
    return data

def downsample_waveform(waveform, n_bins):
    waveforms = np.zeros(n_bins)
    n_points = len(waveform) // n_bins
    for i in range(n_bins):
        start = i * n_points
        end = start + n_points
        waveforms[i] = np.mean(waveform[start:end])
    return waveforms

def convert_spectrogram(data):
    n_bins = 16
    fft_size = 1024
    res = []
    for i in range(0, len(data), fft_size):
        spec = ulab.utils.spectrogram(data[i:i+fft_size])
        spec[0] = 0
        mspec = downsample_waveform(spec, n_bins)
        res.extend(mspec)
        del spec
        del mspec
        gc.collect()
    del data
    gc.collect()
    res = np.array(res)
    amax = np.argmax(res)
    amin = np.argmin(res)
    nres = (res - amin) / (amax - amin)
    del amax
    del amin
    del res
    gc.collect()
    return np.array(nres)

def set_color(led, color):
    led[0] = (0, 0, 0)
    if color.lower() == "red":
        led[0] = (255, 0, 0)
    elif color.lower() == "green":
        led[0] = (0, 255, 0)
    elif color.lower() == "blue":
        led[0] = (0, 0, 255)
    else:
        led[0] = (0, 0, 0)
        
def get_color(r, g, b):
    color_dict = {r: "red", g: "green", b: "blue"}
    return color_dict[max(color_dict.keys())]

#setup led
led = neopixel.NeoPixel(board.NEOPIXEL, 1)
led.brightness = 0.3

#setup mic
mic = analogio.AnalogIn(board.A2)

while True:
    data = read_audio_data(mic)
    gc.collect()
    if np.sum(data) > 1:
        print("data len: {}".format(len(data)))
        spec = convert_spectrogram(data)
        del data
        gc.collect()
        print("spectrogram len: {}".format(len(spec)))
        rscore = red_model.score(spec)[0]
        gc.collect()
        print("Red Score: {}".format(rscore))
        gscore = green_model.score(spec)[0]
        gc.collect()
        print("Green Score: {}".format(gscore))
        bscore = blue_model.score(spec)[0]
        gc.collect()
        print("Blue Score: {}".format(bscore))
        c = get_color(rscore, gscore, bscore)
        print("color: {}".format(c))
        gc.collect()
        del spec
        del rscore
        del gscore
        del bscore
        gc.collect()
        set_color(led, c)
        del c
        gc.collect()
        time.sleep(4)
        print("wait...")
    else:
        del data
        gc.collect()
        print("Free Memory: {}".format(gc.mem_free()))
