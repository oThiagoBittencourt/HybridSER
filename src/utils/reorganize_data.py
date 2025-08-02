# Dataset: The Canadian French Emotional (CaFE),
# licensed under CC BY-NC-SA 4.0. Source: https://zenodo.org/records/1219621

# Dataset: RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song,
# licensed under CC BY-NC-SA 4.0. Source: https://zenodo.org/record/1188976

# Dataset: VERBO - Voice Emotion Recognition dataBase in Portuguese Language,
# licensed under Creative Commons Attribution (CC-BY) 3.0. Source: https://sites.google.com/view/verbodatabase/

import os
import shutil
from pathlib import Path


NEW_PATH = "src//data"
BASE_DIR = "src//data//dataset"


def rename_and_relocate_data(audio_path, language, gender, emotion, id):
    # Renames the audio file and relocates it to a folder based on the emotion.

    # Ensures the existence of the emotion folder
    emotion_dir = os.path.join(NEW_PATH, emotion)
    os.makedirs(emotion_dir, exist_ok=True)

    # Move the audio into the emotion folder (keeping the original name)
    filename = os.path.basename(audio_path)
    dest_path = os.path.join(emotion_dir, filename)
    shutil.move(audio_path, dest_path)

    # Rename the file within the same folder
    new_filename = f"{language}_{gender}_{emotion}_{id}.wav"
    new_path = os.path.join(
        emotion_dir, new_filename
    )  # exemple: "src/data/happy/por_F_happy_0.wav"
    os.rename(dest_path, new_path)

    return


def select_portuguese_labels():
    # Ao through the REVDESS folder, extract id, emotion, gender and audio_path

    PORTUGUESE_EMOTIONS = {
        "ale": "happy",
        "des": "disgust",
        "tri": "sadness",
        "sur": "surprise",
        "rai": "angry",
        "neu": "neutral",
        "med": "fear",
    }
    counters = {}  # Emotion accountant

    # Recursively loops through all .wav files within base_path
    for wav in Path("src//data//dataset//VERBO-Dataset").rglob("*.wav"):
        # Separate the file name by "-"
        parts = wav.stem.split("-")
        if len(parts) != 3:  # If it doesn't have the expected format, skip
            continue

        emo_code, actor_code, _ = parts

        # Select gender
        gender = "F" if actor_code.lower().startswith("f") else "M"

        # Converts emotion code to name
        emotion = PORTUGUESE_EMOTIONS.get(emo_code, "unknown")
        if emotion not in counters:
            counters[emotion] = 0

        # Move and rename the audio
        rename_and_relocate_data(str(wav), "por", gender, emotion, counters[emotion])
        counters[emotion] += 1


def select_french_labels():
    # Ao through the CaFE folder, extract id, emotion, gender and audio_path

    FRENCH_EMOTIONS = {
        "C": "angry",
        "D": "disgust",
        "J": "happy",
        "N": "neutral",
        "P": "fear",
        "S": "surprise",
        "T": "sadness",
    }

    counters = {}  # Emotion accountant

    # Recursively loops through all .wav files within base_path
    for wav in Path("src//data//dataset//CaFE//").rglob("*.wav"):
        # Separate the file name by "-"
        parts = wav.stem.split("-")
        if len(parts) < 3:  # If it doesn't have the expected format, skip
            continue

        actor = int(parts[0])
        emo_code = parts[1]

        # Select gender
        if actor == 11:
            gender = "F"
        elif actor == 12:
            gender = "M"
        else:
            gender = "M" if (actor % 2 != 0) else "F"

        # Converts emotion code to name
        emotion = FRENCH_EMOTIONS.get(emo_code, "unknown")

        if emotion not in counters:
            counters[emotion] = 0

        # Move and rename the audio
        rename_and_relocate_data(str(wav), "por", gender, emotion, counters[emotion])

        counters[emotion] += 1


def select_english_labels():
    # Ao through the REVDESS folder, extract id, emotion, gender and audio_path

    ENGLISH_EMOTIONS = {
        "01": "neutral",
        "02": "Calm",
        "03": "happy",
        "04": "sadness",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    counters = {}  # Emotion accountant

    # Recursively loops through all .wav files within base_path
    for wav in Path("src//data//dataset//REVDESS//Speech//").rglob("*.wav"):
        # Separate the file name by "-"
        parts = wav.stem.split("-")
        if len(parts) != 7:  # If it doesn't have the expected format, skip
            continue

        emo_code = parts[2]
        if emo_code == "02":
            continue  # Ignore the emotion Calm

        actor = int(parts[-1])

        # Select gender
        gender = "M" if (actor % 2 != 0) else "F"

        # Converts emotion code to name
        emotion = ENGLISH_EMOTIONS.get(emo_code, "unknown")

        if emotion not in counters:
            counters[emotion] = 0

        # Move and rename the audio
        rename_and_relocate_data(str(wav), "por", gender, emotion, counters[emotion])
        counters[emotion] += 1


if __name__ == "__main__":
    select_portuguese_labels()
    select_french_labels()
    select_english_labels()
