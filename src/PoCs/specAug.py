import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
from utils.augment import spec_augment

OUTPUT_DIR = "audios/PoCs_Images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar áudio
y, sr = librosa.load("audios/raw/01-C-1-1.wav", sr=None)
print(f"Original: {y.shape}, SR = {sr}")

# Gerar log-Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# Aplicar SpecAugment
augmented_spec = spec_augment(log_mel_spec)

# Salvar imagem do espectrograma
plt.figure(figsize=(10, 4))
specshow(augmented_spec, sr=sr, x_axis='time', y_axis='mel')
plt.title('SpecAugment - Log-Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/specaug.png")
plt.close()

print("Espectrograma com SpecAugment salvo.")
