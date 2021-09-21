from pydub import AudioSegment
'''
t2 = 5000;

newAudio = AudioSegment.from_wav("coughing.wav")
print(len(newAudio))
audio_len = math.floor(len(newAudio)/5000)-1
print(audio_len)

for i in range(audio_len):
    newAudio = AudioSegment.from_wav("coughing.wav")
    t1 = t2
    t2 = t2 + 5000
    newAudio = newAudio[t1:t2]
    newAudio.export('./Done/test/coughing/clip'+str(i)+'.wav',format = "wav")
'''

def slice(name,sec=5):
    strs=name+'.wav'
    a=sec*1000
    newaudio = AudioSegment.from_wav(strs)
    
    for i in range((len(newaudio)//(sec*1000))-1):
        audio = AudioSegment.from_wav(strs)
        b = a
        a = a + 5000
        audio = audio[b:a]
        destination='./Done/test/'+name+'/clip'+str(i)+'.wav'
        audio.export(destination,format = "wav")