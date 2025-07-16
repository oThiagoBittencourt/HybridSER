import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_file = "../files/teste10.wav"
ipd.Audio(audio_file)
signal, sr = librosa.load(audio_file)
print(signal.shape)

mfccs = librosa.feature.mfcc(y = signal, n_mfcc = 13, sr=sr)

#print(mfccs.shape)

delta_mfccs = librosa.feature.delta(data = mfccs)
delta2_mfccs = librosa.feature.delta(data = mfccs, order = 2)

#print(delta_mfccs.shape)
#print(delta2_mfccs.shape)

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(delta2_mfccs, sr=sr, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()

chromagram = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=4096)

plt.figure(figsize=(25, 10))

# librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma')

# plt.colorbar(label='Intensity')
# plt.title('Chromagram')
# plt.tight_layout()
# plt.show()

rms_values = librosa.feature.rms(y=signal)[0]
times = librosa.frames_to_time(np.arange(len(rms_values)), sr=sr)

# plt.figure(figsize=(25, 10))
# plt.plot(times, rms_values)
# plt.title('RMS Energy Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('RMS Energy')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

zcr = librosa.feature.zero_crossing_rate(y=signal)[0]

times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)

# plt.figure(figsize=(25, 10))
# plt.plot(times, zcr)
# plt.title('Zero-Crossing Rate Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('ZCR')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#####  TODO think about how should be the pipeline to transform the datasets 
# into an feedable dataset with all the extractions

#####  See encoder in latency array