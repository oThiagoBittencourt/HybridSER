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


def spec_augment(mel_spec, time_mask_param=30, freq_mask_param=13, num_time_masks=1, num_freq_masks=1):
    augmented = mel_spec.copy()
    num_mel_channels, num_time_steps = augmented.shape

    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(1, num_time_steps - t))
        augmented[:, t0:t0 + t] = 0

    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, max(1, num_mel_channels - f))
        augmented[f0:f0 + f, :] = 0

    return augmented
