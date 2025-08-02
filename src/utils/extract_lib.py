import os
import librosa  # type: ignore
import librosa.display  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from utils import load_config


# --- New Helper Function to Save Spectrogram Kernel ---
def save_spec_as_image(spec_data, file_path, sr, y_axis=None):
    """
    Generates and saves a spectrogram image without axes, borders, or colorbars.
    This function saves only the core content of the plot.

    Args:
        spec_data (np.ndarray): The spectrogram data from librosa.
        file_path (Path or str): The full path to save the image file.
        sr (int): The sampling rate of the audio.
        y_axis (str, optional): The y-axis type for specshow, e.g., 'chroma'.
                                Defaults to None.
    """
    # Create a figure and an axes object. The axes will cover the entire figure.
    fig = plt.figure(figsize=(4, 4), dpi=100)  # DPI can be adjusted for resolution
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()  # Turn off the axes (ticks, labels, spines)
    fig.add_axes(ax)

    # Plot the spectrogram using librosa.display.specshow on the custom axes
    librosa.display.specshow(spec_data, sr=sr, ax=ax, y_axis=y_axis)

    # Save the figure.
    # - bbox_inches='tight' crops the figure to the plotted data.
    # - pad_inches=0 removes any padding around the cropped area.
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0)

    # Close the figure to free up memory
    plt.close(fig)


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
        # Save the kernel of the spectrogram using the new helper function
        save_spec_as_image(delta2_mfccs, target_dir / "dd_mfcc.png", sr)

        # --- 2. Chromagram (Image) ---
        # Calculate features
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sr)
        # Save the kernel, specifying the y_axis type for chromagrams
        save_spec_as_image(
            chromagram, target_dir / "chromagram.png", sr, y_axis="chroma"
        )

        # --- 3. Zero-Crossing Rate (Array) ---
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


# --- Parsing Functions (Unchanged) ---
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
        "01": "Neutral", "02": "Calm", "03": "Joy", "04": "Sad",
        "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised",
    }
    try:
        # FIX: Isolate the numeric identifier before processing
        identifier = name.split('_')[0]

        parts = identifier.split("-")
        if len(parts) != 7:
            return None

        emotion_code = parts[2]
        if emotion_code not in ENGLISH_EMOTIONS:
            return None
        emotion = ENGLISH_EMOTIONS[emotion_code]

        # This will now work on a clean number string like '01', '02', etc.
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

    config = load_config()
    SOURCE_AUDIO_DIRECTORY = config["DATASET_FOLDER"]
    OUTPUT_FEATURES_DIRECTORY = config["OUTPUT_FOLDER_RAW_FEATURES"]

    # --- To run the full pipeline ---
    # 1. Extract features
    # print("--- Starting Feature Extraction ---")
    # process_dataset(SOURCE_AUDIO_DIRECTORY, OUTPUT_FEATURES_DIRECTORY)
    # print("\n--- Feature extraction complete! ---")

    # 2. Rename directories
    print("\n--- Starting Directory Renaming ---")
    rename_feature_directories(OUTPUT_FEATURES_DIRECTORY)
