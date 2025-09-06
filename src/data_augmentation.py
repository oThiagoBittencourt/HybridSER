import os
import librosa
import numpy as np
import random
import soundfile as sf
from utils.audio_metadado import get_audio_metadado


SNR_LOW = [23, 30]
SNR_HIGH = [16, 23]

RATE_LOW = [[1, 1.04], [0.96, 1]]  # RATE [[speed up], [slow down]]
RATE_HIGH = [[1.04, 1.1], [0.9, 0.96]]

N_STEPS_FEMALE = [[2, 3], [1, 2]]  # N_STEPS [[high], [low]]
N_STEPS_MALE = [[-3, -2], [-2, -1]]

COMBINATIONS = [
    "pitch",
    "time",
    "awgn",
    "pitch_time",
    "pitch_awgn",
    "time_awgn",
    "pitch_time_awgn",
]


def get_snr(intensity):
    """Returns random SNR value based on intensity level"""
    if intensity == 1:
        return random.randint(SNR_HIGH[0], SNR_HIGH[1])
    return random.randint(SNR_LOW[0], SNR_LOW[1])


def get_rate(intensity):
    """Returns random rate value for time stretching based on intensity level"""
    if intensity == 1:
        return random.choice([
            random.uniform(RATE_HIGH[0][0], RATE_HIGH[0][1]),
            random.uniform(RATE_HIGH[1][0], RATE_HIGH[1][1]),
        ])
    return random.choice([
        random.uniform(RATE_LOW[0][0], RATE_LOW[0][1]),
        random.uniform(RATE_LOW[1][0], RATE_LOW[1][1]),
    ])


def get_steps(gender, intensity):
    """Returns pitch shift steps based on gender and intensity level"""
    if gender == "F":
        if intensity == 1:
            return random.uniform(N_STEPS_FEMALE[0][0], N_STEPS_FEMALE[0][1])
        return random.uniform(N_STEPS_FEMALE[1][0], N_STEPS_FEMALE[1][1])
    if intensity == 1:
        return random.uniform(N_STEPS_MALE[0][0], N_STEPS_MALE[0][1])
    return random.uniform(N_STEPS_MALE[1][0], N_STEPS_MALE[1][1])


def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)


def awgn(audio, snr_db):
    snr = 10 ** (snr_db / 10)
    power = np.mean(audio**2)
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def aplly_transforms(audio, sr, gender, intensity, combination):
    """Applies sequence of audio transformations based on specified combination"""
    audio_transformed = audio.copy()
    transformations = combination.split("_")

    for transform in transformations:
        if transform == "pitch":
            n_steps = get_steps(gender, intensity)
            audio_transformed = pitch_shift(audio_transformed, sr, n_steps)
        elif transform == "time":
            rate = get_rate(intensity)
            audio_transformed = time_stretch(audio_transformed, rate)
        elif transform == "awgn":
            snr_db = get_snr(intensity)
            audio_transformed = awgn(audio_transformed, snr_db)

    return audio_transformed


def process_directory(directory_source):
    """
    Processes all WAV audio files in the source directory and generates augmented versions.
    For each audio file, applies multiple transformation combinations at different intensity levels,
    organizes output by emotion categories, and saves with descriptive filenames containing metadata.
    """
    directory_augmented = os.path.join("src//data", "augmented")
    os.makedirs(directory_augmented, exist_ok=True)

    count_id = 0

    files_wav = []
    for root, dirs, files in os.walk(directory_source):
        for file in files:
            if file.endswith(".wav"):
                complete_path = os.path.join(root, file)
                files_wav.append(complete_path)

    for path_file in files_wav:
        file = os.path.basename(path_file)

        try:
            meta = get_audio_metadado(file)

            emotion_dir = os.path.join(directory_augmented, meta.emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            audio, sr = librosa.load(path_file)

            for intensity in [0, 1]:
                for combination in COMBINATIONS:
                    audio_augmented = aplly_transforms(
                        audio, sr, meta.gender, intensity, combination
                    )

                    new_id = f"{meta.id}_{count_id}"
                    new_name = f"{meta.language}_{meta.gender}_{meta.emotion}_{new_id}_{intensity}_{combination}.wav"
                    new_path = os.path.join(emotion_dir, new_name)

                    sf.write(new_path, audio_augmented, sr)

                    count_id += 1

        except ValueError as e:
            print(f"Error {file}: {e}")
            continue

    print("Data Augmentation ended...")


if __name__ == "__main__":
    directory_source = "src//data//dataset"
    process_directory(directory_source)
