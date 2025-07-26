import os
import numpy as np
import librosa
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import tensorflow as tf


def run_audio_cnn_poc():
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory setup
    base_dir = os.path.dirname(__file__)
    audio_path = os.path.join(base_dir, '..', 'data', 'example.wav')

    # Load and normalize audio with librosa (resamples to 16kHz)
    try:
        audio, sr = librosa.load(audio_path, sr=16000)  # sr=16000 forces resampling
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")
    print("Processing:", audio_path)

    audio = audio.astype(np.float32)  # librosa already normalizes to [-1,1]
    audio = audio[:16000] if len(audio) >= 16000 else np.pad(audio, (0, 16000 - len(audio)))
    audio = np.expand_dims(audio, axis=(0, 2))  # shape: (1, time, 1)

    # Layer definition + manual forward pass
    conv1 = Conv1D(64, 3, activation='relu')
    pool1 = MaxPooling1D(2)
    conv2 = Conv1D(128, 3, activation='relu')
    pool2 = MaxPooling1D(2)
    flatten = Flatten()

    x = conv1(audio)
    print("After conv1:", x.shape)
    x = pool1(x)
    print("After pool1:", x.shape)
    x = conv2(x)
    print("After conv2:", x.shape)
    x = pool2(x)
    print("After pool2:", x.shape)
    flat_out = flatten(x)

    # Outputs
    print("Flatten shape:", flat_out.shape)
    print("First 10 values:", flat_out[0, :10].numpy())


if __name__ == "__main__":
    run_audio_cnn_poc()

"""
Possible future improvements:

Preprocessing:
   - Normalize audio to [-1,1] (already done).
   - Standardize audio length (e.g., 16000 samples).

Network architecture:
   - Adjust number of filters and kernel sizes.
   - Add Batch Normalization after Conv1D layers for stability.
   - Add Dropout to prevent overfitting.
   - Try different activation functions.
"""
