import soundfile as sf
import librosa
from utils.augment import add_awgn
import os

OUTPUT_DIR = "audios/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar áudio
y, sr = librosa.load("audios/03-01-02-01-01-02-01.wav", sr=None)
print(f"Original: {y.shape}, SR = {sr}")

# Aplicar ruído
y_awgn = add_awgn(y, snr_db=20)

# Salvar saída
sf.write(f"{OUTPUT_DIR}/awgn.wav", y_awgn, sr)
print("Arquivo com AWGN salvo.")
