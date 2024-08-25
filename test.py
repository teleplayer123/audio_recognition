import winsound as ws
import os
import pyaudio
import wave

down_dirs = os.path.join(os.getcwd(), "data", "mini_speech_commands", "down")
file_list = [os.path.join(down_dirs, fname) for fname in os.listdir(down_dirs)]

f0 = file_list[1]

# ws.PlaySound(f0, ws.SND_FILENAME)



# Set chunk size of 1024 samples per data frame
chunk = 1024  

# Open the sound file 
wf = wave.open(f0, 'rb')

# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)

# Play the sound by writing the audio data to the stream
while True:
    if data == "" or len(data) < 1:
        break
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()