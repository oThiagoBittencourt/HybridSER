import soundfile as sf
import librosa
from utils.augment import time_stretch, tranform_estereo_to_mono
import os

OUTPUT_DIR = "audios/PoCs_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

y, sr = librosa.load("audios/raw/01-C-1-1.wav", sr=None)
y = tranform_estereo_to_mono(y)
print(f"Original: {y.shape}, SR = {sr}")

y_time = time_stretch(y, rate=0.6)

sf.write(f"{OUTPUT_DIR}/time.wav", y_time, sr)
print("Arquivo com time stretch salvo.")
