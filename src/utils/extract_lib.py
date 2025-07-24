import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils import load_config


def extract_features(audio_path, output_dir):
    """
    Extracts features from an audio file and saves them to a specified directory.

    For each audio file, it creates a subdirectory in output_dir named after the
    audio file. Inside this subdirectory, it saves:
    - Delta-Delta MFCCs as an image (dd_mfcc.png)
    - Chromagram as an image (chromagram.png)
    - Zero-Crossing Rate as a NumPy array (zcr.npy)
    - RMS Energy as a NumPy array (rms.npy)
    """
    try:
        file_stem = Path(audio_path).stem
        target_dir = Path(output_dir) / file_stem
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {audio_path} -> {target_dir}")

        signal, sr = librosa.load(audio_path, sr=None)

        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
        delta2_mfccs = librosa.feature.delta(data=mfccs, order=2)

        plt.figure(figsize=(12, 8))
        librosa.display.specshow(delta2_mfccs, sr=sr, x_axis="time")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Delta-Delta MFCCs\n({file_stem})")
        plt.tight_layout()
        plt.savefig(target_dir / "dd_mfcc.png")
        plt.close()

        chromagram = librosa.feature.chroma_stft(y=signal, sr=sr)

        plt.figure(figsize=(12, 8))
        librosa.display.specshow(chromagram, sr=sr, x_axis="time", y_axis="chroma")
        plt.colorbar(label="Intensity")
        plt.title(f"Chromagram\n({file_stem})")
        plt.tight_layout()
        plt.savefig(target_dir / "chromagram.png")
        plt.close()

        zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
        np.save(target_dir / "zcr.npy", zcr)

        rms = librosa.feature.rms(y=signal)[0]
        np.save(target_dir / "rms.npy", rms)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def process_dataset(
    source_root, output_root, audio_extensions=(".wav", ".mp3", ".flac")
):
    """
    Recursively finds all audio files in the source_root, extracts their
    features, and saves them to the output_root.
    """
    Path(output_root).mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_path = os.path.join(root, file)
                extract_features(audio_path, output_root)


def parse_french_name(name):
    """Attempts to parse a French filename pattern. Returns new name or None."""
    FRENCH_EMOTIONS = {
        "C": "Anger",
        "D": "Disgust",
        "J": "Joy",
        "N": "Neutral",
        "P": "Fear",
        "S": "Surprise",
        "T": "Sadness",
    }
    try:
        parts = name.split("-")
        if len(parts) < 2:
            return None

        speaker_id = int(parts[0])
        gender = "M" if speaker_id % 2 != 0 else "F"

        emotion_code = parts[1].upper()
        if emotion_code not in FRENCH_EMOTIONS:
            return None
        emotion = FRENCH_EMOTIONS[emotion_code]

        return f"fr_{gender}_{emotion}"
    except (ValueError, IndexError):
        return None


def parse_english_name(name):
    """Attempts to parse an English filename pattern. Returns new name or None."""

    ENGLISH_EMOTIONS = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised",
    }
    try:
        parts = name.split("-")
        if len(parts) != 7:
            return None

        emotion_code = parts[2]
        if emotion_code not in ENGLISH_EMOTIONS:
            return None
        emotion = ENGLISH_EMOTIONS[emotion_code]

        actor_id = int(parts[6])
        gender = "M" if actor_id % 2 != 0 else "F"

        if emotion == "Fearful":
            emotion = "Fear"

        return f"eng_{gender}_{emotion}"
    except (ValueError, IndexError):
        return None


def parse_portuguese_name(name):
    """Attempts to parse a Portuguese filename pattern. Returns new name or None."""

    PORTUGUESE_EMOTIONS = {
        "ale": "Joy",
        "des": "Disgust",
        "tri": "Sadness",
        "sur": "Surprise",
        "rai": "Anger",
        "neu": "Neutral",
        "med": "Fear",
    }
    try:
        parts = name.split("-")
        if len(parts) < 2:
            return None

        emotion_code = parts[0].lower()
        if emotion_code not in PORTUGUESE_EMOTIONS:
            return None
        emotion = PORTUGUESE_EMOTIONS[emotion_code]

        gender_code = parts[1][0].lower()
        if gender_code == "f":
            gender = "F"
        elif gender_code == "m":
            gender = "M"
        else:
            return None

        return f"por_{gender}_{emotion}"
    except IndexError:
        return None


# --- Main Processing Function ---


def rename_feature_directories(root_dir):
    """
    Walks through a directory and renames subdirectories based on
    French, English, or Portuguese naming patterns.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Directory not found at '{root_dir}'")
        return

    unparsed_dirs = []
    parsers = [parse_french_name, parse_english_name, parse_portuguese_name]

    print(f"Scanning directories in: {root_path.resolve()}\n")

    for path in root_path.iterdir():
        if not path.is_dir():
            continue

        original_name = path.name
        new_base_name = None

        for parser in parsers:
            result = parser(original_name)
            if result:
                new_base_name = result
                break

        if new_base_name:
            counter = 1
            new_path = root_path / new_base_name
            while new_path.exists():
                new_name_with_counter = f"{new_base_name}_{counter}"
                new_path = root_path / new_name_with_counter
                counter += 1

            try:
                path.rename(new_path)
                print(f"Renamed: '{original_name}' -> '{new_path.name}'")
            except OSError as e:
                print(f"ERROR renaming '{original_name}': {e}")
        else:
            unparsed_dirs.append(original_name)
            print(f"Skipped: '{original_name}' (no matching pattern found)")

    print("\n--- Renaming complete! ---")
    if unparsed_dirs:
        print("\nThe following directories could not be parsed:")
        for name in unparsed_dirs:
            print(f"- {name}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Testing packages and scripts")

    config = load_config()

    print(config["DATASET_FOLDER"])
    print(config["OUTPUT_FOLDER_RAW_FEATURES"])
    # SOURCE_AUDIO_DIRECTORY = 'C:/Users/thiago/OneDrive/Desktop/TCC/Code/datasets'
    # OUTPUT_FEATURES_DIRECTORY = 'C:/Users/thiago/OneDrive/Desktop/TCC/Code/data'

    # Run the processing pipeline
    # process_dataset(SOURCE_AUDIO_DIRECTORY, OUTPUT_FEATURES_DIRECTORY)

    # print("\n--- Feature extraction complete! ---")
    # print(f"All features saved in: {OUTPUT_FEATURES_DIRECTORY}")

    # root_dir = Path("C:/Users/thiago/OneDrive/Desktop/TCC/Code/data")
    # rename_feature_directories(root_dir)
