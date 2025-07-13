import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt


audio_file = "../files/teste10.wav"
ipd.Audio(audio_file)
signal, sr = librosa.load(audio_file)
print(signal.shape)

mfccs = librosa.feature.mfcc(y = signal, n_mfcc = 13, sr=sr)

print(mfccs.shape)

plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
#plt.show()

delta_mfccs = librosa.feature.delta(data = mfccs)
delta2_mfccs = librosa.feature.delta(data = mfccs, order = 2)

print(delta_mfccs.shape)
print(delta2_mfccs.shape)
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(delta_mfccs, sr=sr, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# #plt.show()

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(delta2_mfccs, sr=sr, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# #plt.show()