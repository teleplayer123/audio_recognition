import time
import board
import analogio
import adafruit_displayio_ssd1306
from adafruit_bitmap_font import bitmap_font
from adafruit_display_text import label
import displayio
from ulab import numpy as np
import ulab
import gc
import red_model


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
    amax = np.max(res)
    amin = np.min(res)
    nres = (res - amin) / (amax - amin)
    del amax
    del amin
    del res
    gc.collect()
    return np.array(nres)

displayio.release_displays()

i2c = board.STEMMA_I2C()
width = 128
height = 64
# i2c = busio.I2C(scl, sda)
display_bus = displayio.I2CDisplay(i2c, device_address=0x3C)
display = adafruit_displayio_ssd1306.SSD1306(display_bus, width=128, height=64)

font = bitmap_font.load_font("./Helvetica-Bold-16.bdf")

splash = displayio.Group()
display.root_group = splash

def add_border():
    global splash
    #create base rectangle filled white
    color_bitmap = displayio.Bitmap(128, 64, 1)
    color_palette = displayio.Palette(1)
    color_palette[0] = 0xFFFFFF #white
    bg_sprite = displayio.TileGrid(color_bitmap, pixel_shader=color_palette, x=0, y=0)
    splash.append(bg_sprite)

    #creates inner rectangle filled black
    inner_bitmap = displayio.Bitmap(126, 62, 1)
    inner_palette = displayio.Palette(1)
    inner_palette[0] = 0x000000  #black
    inner_sprite = displayio.TileGrid(inner_bitmap, pixel_shader=inner_palette, x=1, y=1)
    splash.append(inner_sprite)

def add_label():
    global splash
    #creates label with text audio
    audio_label = label.Label(font, text="Audio", color=0xFFFFFF)
    audio_label.anchor_point = (0.5, 0.0) #centers label text
    audio_label.anchored_position = (64, 5) #positions label on x,y axis above value
    splash.append(audio_label)


#creates text oject to write values
audio_string = ""
audio_value_label = label.Label(font, text=audio_string, color = 0xFFFFFF)
audio_value_label.anchor_point = (0.5, 0.5) #centers text
audio_value_label.anchored_position = (64, 38) #positions value text on x,y axis below label
audio_value_label.scale = 3
splash.append(audio_value_label)

def display_text(val):
    global audio_value_label
    audio_string = "{}".format(val)
    audio_value_label.text = audio_string

#setup the microphone
mic = analogio.AnalogIn(board.A0)

gc.collect()
print("Free Memory: {}".format(gc.mem_free()))

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
        display_text(rscore)
        gc.collect()