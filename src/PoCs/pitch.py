import soundfile as sf
import librosa
from utils.augment import pitch_shift
import os

OUTPUT_DIR = "audios/PoCs_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

y, sr = librosa.load("audios/raw/01-C-1-1.wav", sr=None)
print(f"Original: {y.shape}, SR = {sr}")

y_pitch = pitch_shift(y, sr, steps=2)

sf.write(f"{OUTPUT_DIR}/pitch.wav", y_pitch, sr)
print("Arquivo com pitch shift salvo.")
