# Dataset: The Canadian French Emotional (CaFE),
# licensed under CC BY-NC-SA 4.0. Source: https://zenodo.org/records/1219621

# Dataset: RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song,
# licensed under CC BY-NC-SA 4.0. Source: https://zenodo.org/record/1188976

# Dataset: VERBO - Voice Emotion Recognition dataBase in Portuguese Language,
# licensed under Creative Commons Attribution (CC-BY) 3.0. Source: https://sites.google.com/view/verbodatabase/

from pathlib import Path
from collections import defaultdict
import soundfile as sf
import numpy as np
import librosa

NEW_PATH = "src//data//dataset"


def transform_stereo_to_mono(y):
    """
    Converts stereo audio signal to mono by averaging channels.
    """
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    return y


def load_as_mono(path):
    """
    Loads audio file as mono signal with automatic stereo-to-mono conversion.
    """
    info = sf.info(str(path))

    y, sr = librosa.load(path, sr=None, mono=info.channels == 1)

    if info.channels > 1:
        try:
            y = transform_stereo_to_mono(y)
        except Exception:
            y = np.mean(y, axis=0) if y.ndim > 1 else y

    return y.astype(np.float32), sr


def rename_and_relocate_data(audio_path, language, gender, emotion, id):
    """
    Moves, renames, and converts audio files to mono format with standardized naming.
    """
    audio_path = Path(audio_path)

    y, sr = load_as_mono(audio_path)

    emotion_dir = Path(NEW_PATH) / emotion
    emotion_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{language}_{gender}_{emotion}_{id}.wav"
    new_path = emotion_dir / base_name

    if new_path.exists():
        dup = 1
        while True:
            new_name = f"{language}_{gender}_{emotion}_{id}_dup{dup}.wav"
            new_path = emotion_dir / new_name
            if not new_path.exists():
                break
            dup += 1

    sf.write(new_path, y, sr)
    audio_path.unlink()

    return str(new_path)


def select_portuguese_labels():
    """
    Processes Portuguese audio dataset (VERBO-Dataset).
    Organizes files by language, gender, and emotion according to predefined mapping.
    """
    PORTUGUESE_EMOTIONS = {
        "ale": "happy",
        "des": "disgust",
        "tri": "sadness",
        "sur": "surprise",
        "rai": "angry",
        "neu": "neutral",
        "med": "fear",
    }

    base = Path("src//data//dataset_base//VERBO-Dataset//Audios")

    if not base.exists():
        return

    groups = defaultdict(list)

    for wav in base.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 2:
            continue
        emo_code = parts[0]
        actor_code = parts[1]

        gender = "F" if actor_code.lower().startswith("f") else "M"
        emotion = PORTUGUESE_EMOTIONS.get(emo_code, "unknown")

        key = ("por", gender, emotion)
        groups[key].append(wav)

    for key in sorted(groups.keys()):
        lang, gender, emotion = key
        files = sorted(groups[key], key=lambda p: p.name)
        for idx, wav in enumerate(files, start=1):
            rename_and_relocate_data(str(wav), lang, gender, emotion, idx)


def select_french_labels():
    """
    Processes French audio dataset (CaFE).
    Organizes files by language, gender, and emotion according to predefined mapping.
    """
    FRENCH_EMOTIONS = {
        "C": "angry",
        "D": "disgust",
        "J": "happy",
        "N": "neutral",
        "P": "fear",
        "S": "surprise",
        "T": "sadness",
    }

    base = Path("src//data//dataset_base//CaFE")

    if not base.exists():
        return

    groups = defaultdict(list)

    for wav in base.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        try:
            actor = int(parts[0])
        except ValueError:
            continue
        emo_code = parts[1]

        # actor 11 is female, 12 male
        if actor == 11:
            gender = "F"
        elif actor == 12:
            gender = "M"
        else:
            gender = "M" if (actor % 2 != 0) else "F"

        emotion = FRENCH_EMOTIONS.get(emo_code, "unknown")

        key = ("fra", gender, emotion)
        groups[key].append(wav)

    for key in sorted(groups.keys()):
        lang, gender, emotion = key
        files = sorted(groups[key], key=lambda p: p.name)
        for idx, wav in enumerate(files, start=1):
            rename_and_relocate_data(str(wav), lang, gender, emotion, idx)


def select_english_labels():
    """
    Processes English audio dataset (REVDESS).
    Organizes files by language, gender, and emotion according to predefined mapping.
    Skips 'calm' emotion category during processing.
    """
    ENGLISH_EMOTIONS = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sadness",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }

    base = Path("src//data//dataset_base//REVDESS//Speech")

    if not base.exists():
        return

    groups = defaultdict(list)

    for wav in base.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue

        emo_code = parts[2]
        if emo_code == "02":
            continue

        try:
            actor = int(parts[-1])
        except ValueError:
            continue

        gender = "M" if (actor % 2 != 0) else "F"
        emotion = ENGLISH_EMOTIONS.get(emo_code, "unknown")

        key = ("eng", gender, emotion)
        groups[key].append(wav)

    for key in sorted(groups.keys()):
        lang, gender, emotion = key
        files = sorted(groups[key], key=lambda p: p.name)
        for idx, wav in enumerate(files, start=1):
            rename_and_relocate_data(str(wav), lang, gender, emotion, idx)


if __name__ == "__main__":
    select_portuguese_labels()
    select_french_labels()
    select_english_labels()
