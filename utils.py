import winsound as ws
import os
import pyaudio
import wave


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

def play_wavedir(path):
    file_list = [os.path.join(path, fname) for fname in os.listdir(path)]
    for wav in file_list:
        ws.PlaySound(wav, ws.SND_FILENAME)

