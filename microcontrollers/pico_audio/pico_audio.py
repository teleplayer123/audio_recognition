import math
import board
import analogio
from adafruit_ssd1306 import SSD1306_I2C
import adafruit_displayio_ssd1306
from adafruit_bitmap_font import bitmap_font
from adafruit_display_text import label
import displayio
from ulab import numpy as np


# Helper functions to remove DC bias and compute RMS
def mean(values):
    return sum(values) / len(values)

def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(float(sample - minbuf) * (sample - minbuf) for sample in values)
    return math.sqrt(samples_sum / len(values))


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

def add_border(splash):
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

def add_label(splash):
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

#setup the microphone
mic = analogio.AnalogIn(board.A0)

while True:
    val = mic.value
    audio_string = "{}".format(val)
    audio_value_label.text = audio_string
    print(val)
