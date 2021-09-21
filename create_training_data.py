import numpy as np
import os
from tqdm import tqdm
import random
import pickle
from scipy.io.wavfile import read
import slice
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display

slice.slice('coughing')
slice.slice('not_coughing')

datadir = "Done/test/"

classes = ["coughing","not_coughing"]
train_data= []

def create_training_data():
    for category in classes:

        y = classes.index(category)

        for audio in tqdm(os.listdir(datadir+category)):
            global input_data
            samplerate,input_data = read(datadir+category+'/'+audio)
            #print( samplerate,len(input_data))
            train_data.append([input_data,y])
            
            
create_training_data()    



print((train_data[0][0])[:])
print((train_data[0])[:])

print(len(train_data))
random.shuffle(train_data)



#for AUDIO PLOT 
sample_rate, samples = wavfile.read('Done/test\coughing\clip0.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('Done/test\coughing\clip1.wav')
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, data, color='blue') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.show()

x, sr = librosa.load('Done/test\coughing\clip3.wav')
print(type(x), type(sr))
print(x.shape, sr)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()










X = []
y = []



for samples,label in train_data:
    X.append(samples)
    y.append(label)


X = np.array(X).reshape(73, 1, 220500, -1)

print(X.shape)
print(y)
print(len(y))

with open('picklefiles/X.pickle', 'wb') as f:
    pickle.dump(X, f)
    f.close()
with open('picklefiles/y.pickle', 'wb') as f:
    pickle.dump(y, f)
    f.close()


