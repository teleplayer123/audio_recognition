import array
import board
import audiobusio
import displayio
from adafruit_gizmo import tft_gizmo
from ulab import numpy as np
import ulab
import time
import gc


display = tft_gizmo.TFT_Gizmo()

# Create a heatmap color palette
palette = displayio.Palette(52)
# fmt: off
for i, pi in enumerate((0xff0000, 0xff0a00, 0xff1400, 0xff1e00,
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
                        0x0022ff, 0x0012ff, 0x0002ff, 0x0000ff)):
    # fmt: on
    palette[51-i] = pi

class RollingGraph(displayio.TileGrid):
    def __init__(self, scale=2):
        # Create a bitmap with heatmap colors
        self._bitmap = displayio.Bitmap(display.width//scale,
                                       display.height//scale, len(palette))
        super().__init__(self._bitmap, pixel_shader=palette)

        self.scroll_offset = 0

    def show(self, data):
        y = self.scroll_offset
        bitmap = self._bitmap
        display.auto_refresh = False
        offset = max(0, (bitmap.width-len(data))//2)
        for x in range(min(bitmap.width, len(data))):
            bitmap[x+offset, y] = int(data[x])

        display.auto_refresh = True

        self.scroll_offset = (y + 1) % self.bitmap.height

group = displayio.Group(scale=3)
graph = RollingGraph(3)
fft_size = 256

group.append(graph)
display.root_group = group


mic = audiobusio.PDMIn(board.MICROPHONE_CLOCK, board.MICROPHONE_DATA,
                       sample_rate=16000, bit_depth=16)

def mean(values):
    return sum(values) / len(values)

def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(float(sample - minbuf) * (sample - minbuf) for sample in values)
    return np.sqrt(samples_sum / len(values))

def read_audio_data(mic):
    n_samples = 8192
    data = array.array("H", [0] * n_samples)
    mic.record(data, n_samples)
    return np.array(data)

def get_spectrogram(data):
    spect = ulab.utils.spectrogram(data)
    spect[0] = 0
    return spect

def downsample_waveform(waveform, n_bins):
    waveforms = np.zeros(n_bins)
    waveform = np.array(waveform)
    n_points = len(waveform) // n_bins
    for i in range(n_bins):
        start = i * n_points
        end = start + n_points
        waveforms[i] = np.mean(waveform[start:end])
    return waveforms

def convert_spectrogram(data):
    orig_len = 8192 // 32
    n_bins = 16
    fft_size = 32
    res = []
    for i in range(0, orig_len, fft_size):
        spec = np.fft.fft(data[i*fft_size:i*fft_size+fft_size])
        mspec = downsample_waveform(spec, n_bins)
        res.extend(mspec)
    return np.array(res)

while True:
    data = read_audio_data(mic)
    if len(data) >= 8192:
        print(len(data))
        spec = convert_spectrogram(data)