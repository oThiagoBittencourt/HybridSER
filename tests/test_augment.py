import numpy as np
from src.utils.augment import add_awgn, pitch_shift, time_stretch

def generate_dummy_audio(duration_sec=1, sr=16000):
    return np.ones(sr * duration_sec), sr

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
