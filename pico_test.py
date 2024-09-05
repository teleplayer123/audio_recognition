from machine import Pin, PWM, ADC, SPI
import math
import time
from ulab import numpy as np
import ulab
import sdcard
import uos
import gc


CS = Pin(13, Pin.OUT)
SCK = Pin(10)
MISO = Pin(12)
MOSI = Pin(11)

spi = SPI(1, baudrate=1000000, polarity=0, phase=0, bits=8, firstbit=SPI.MSB, sck=SCK, mosi=MOSI, miso=MISO)
sd = sdcard.SDCard(spi, CS)
vfs = uos.VfsFat(sd)
uos.mount(vfs, "/sd")
        
def read_audio_data(a):
    data = []
    n_samples = 2048
    n_step = 16
    for i in range(0, n_samples * n_step, n_step):
        data.append(100*((a.read_u16() * 3.3 / 65536)) - 1.65)
    if len(data) % 2 != 0:
        data = data[1:]
    return np.array(data)

def get_spectrogram(data):
    spec = ulab.utils.spectrogram(data)
    spect[0] = 0
    return spect

def avg_spectrogram(data, n_bins):
    n_chunks = len(data) // n_bins
    res = [np.mean(data[i*n_bins:(i+1)*nbins]) for i in range(n_chunks)]
    return res

def convert_spectrogram(data):
    n_bins = 16
    fft_size = 1024
    res = []
    for i in range(0, len(data), fft_size):
        start = i * fft_size
        end = start + fft_size
        chunk = data[start:end]
        if len(chunk) != fft_size:
            continue
        spec = get_spectrogram(chunk[:len(chunk)//2])
        mspec = avg_spectrogram(spec, n_bins)
        res.extend(mspec)
    res = np.array(res)
    np.savetxt("/sd/audio1.txt", res)
    res_min = np.min(res)
    res_max = np.max(res)
    res = (res - res_min) / (res_max - res_min)
    return res
        
    

def set_color(color):
    blue = PWM(Pin(0), freq=1000)
    green = PWM(Pin(1), freq=1000)
    red = PWM(Pin(2), freq=1000)
    if color.lower() == "red":
        red.duty_u16(65535)
        green.duty_u16(0)
        blue.duty_u16(0)
    elif color.lower() == "green":
        red.duty_u16(0)
        green.duty_u16(65535)
        blue.duty_u16(0)
    elif color.lower() == "blue":
        red.duty_u16(0)
        green.duty_u16(0)
        blue.duty_u16(65535)
    elif color.lower() == "off":
        red.duty_u16(0)
        green.duty_u16(0)
        blue.duty_u16(0)
    else:
        print("color must be one of: red, green, blue, or off")

def pulse(p, data):
    for i in range(len(data)):
        p.duty_u16(int(math.sin(i / 10 * math.pi) * 5000 + 5000))
        time.sleep_ms(50)

# p0 = PWM(Pin(0), freq=1000)
# p1 = PWM(Pin(1), freq=1000)
# p2 = PWM(Pin(2), freq=1000)

a0 = ADC(Pin(26))

data = read_audio_data(a0)

# np.savetxt("/sd/audio1.txt", data)
    
    

