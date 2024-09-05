from machine import Pin, PWM, ADC, SPI
import math
import time
from ulab import numpy as np
import sdcard
import uos


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
    start = time.time()
    while True:
        data.append(a.read_u16())
        end = time.time()
        if end - start >= 1:
            break
    return data

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

with open("/sd/audio1.txt", "w") as fh:
    fh.write(str(data))
    
    

