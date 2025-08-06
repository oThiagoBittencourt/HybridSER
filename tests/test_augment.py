import numpy as np
from src.utils.augment import add_awgn, pitch_shift, time_stretch, spec_augment


def generate_dummy_audio(duration_sec=1, sr=16000):
    return np.ones(sr * duration_sec), sr


def generate_dummy_spectrogram(n_mels=128, n_frames=100):
    return np.ones((n_mels, n_frames), dtype=np.float32)


def test_add_awgn():
    y, _ = generate_dummy_audio()
    y_noisy = add_awgn(y)
    assert y_noisy.shape == y.shape
    assert not np.array_equal(y, y_noisy)


def test_pitch_shift():
    y, sr = generate_dummy_audio()
    y_pitch = pitch_shift(y, sr)
    assert y_pitch.shape == y.shape


def test_time_stretch():
    y, _ = generate_dummy_audio()
    y_stretched = time_stretch(y, rate=0.9)
    assert y_stretched.shape[0] != y.shape[0]


def test_spec_augment():
    spec = generate_dummy_spectrogram()
    augmented_spec = spec_augment(spec)

    assert augmented_spec.shape == spec.shape, "Dimensão do espectrograma deve permanecer igual"
    assert not np.array_equal(spec, augmented_spec), "A saída do specAugment não deve ser idêntica à entrada"
