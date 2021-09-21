from pydub import AudioSegment


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