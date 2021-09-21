import pyaudio
import wave
import tensorflow as tf
from scipy.io.wavfile import read
import numpy as np
import sys


CATEGORIES = ["cough", "not cough"]
model = tf.keras.models.load_model('cough_or_not.h5')


CHUNK = 4410
FORMAT = pyaudio.get_format_from_width(2, unsigned=False)
CHANNELS = 1
RATE = 88200

WAVE_OUTPUT_FILENAME = "output.wav"
#print(FORMAT)


def is_static(arr):
    return max(arr.flatten()) < 5000


while True:
    try:
        p = pyaudio.PyAudio()
    
        stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
        print('.')
        print('.')
        print("--------listening--------")
        print('.')
        print('.')
        
        frames = []
        
        for i in range(0, 100):
            data = stream.read(CHUNK)
            frames.append(data)
        #print(len(data))
        #print(len(frames))
        print("-------testing---------")
        print('.')
        print('.')
        print('.')
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()    
        wav_rate, audio = read('./output.wav')
        print(wav_rate,len(audio))
    
        X = np.array(audio).reshape(2, 1, 220500, 1)
    
        if(is_static(X[0])):
            print('not enough sound /    static')
        else:
            prediction = model.predict(X)[0][0].item()
            print(CATEGORIES[round(prediction)],prediction)
    except:
        print('system interrupted')
        sys.exit()