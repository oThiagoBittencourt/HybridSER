import librosa
import numpy as np


def add_awgn(y, snr_db=20):
    rms = np.sqrt(np.mean(y**2))
    snr = 10 ** (snr_db / 10)
    noise_std = rms / np.sqrt(snr)
    noise = np.random.normal(0, noise_std, y.shape[0])
    return y + noise


def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def tranform_estereo_to_mono(y):
    if y.ndim == 2:
        print(f"Convertendo estéreo para mono (shape original: {y.shape})")
        y = librosa.to_mono(y.T)
        print(f"Novo shape: {y.shape}")
    return y
