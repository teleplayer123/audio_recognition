# import array
# import board
# import audiobusio
# import displayio
# from adafruit_gizmo import tft_gizmo
# from ulab import numpy as np
# import ulab

xs = (0xff0000, 0xff0a00, 0xff1400, 0xff1e00,
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

s = "testing"

for x in xs:
    t = set_text_color(x, s)
    print(t)