import librosa

audio, fs = librosa.load("datasets/test/clean/clean_fileid_9.wav", sr=None)

print(fs)