import os
import shutil
from collections import defaultdict
from utils.audio_metadado import get_audio_metadado

AUDIO_DICT = defaultdict(lambda: defaultdict(int))  # Count of audio files by language and emotion
EMOTION_FILES = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # File details by language, emotion, and gender
DUPLICATION_INDICES = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Track which files have been duplicated
REVERSE_ORDER = defaultdict(lambda: defaultdict(bool))  # Alternate gender selection order for balancing
MAX_IDS = defaultdict(lambda: defaultdict(int))  # Track maximum ID for each language-emotion combination


def read_data():
    """
    Reads all WAV files from the dataset directory and populates the global data structures.
    Extracts metadata from each audio file and organizes it by language, emotion, and gender.
    """
    for root, directories, files in os.walk("src//data//dataset"):
        for file in files:
            if file.lower().endswith(".wav"):

                try:
                    meta = get_audio_metadado(file)

                    AUDIO_DICT[meta.language][meta.emotion] += 1

                    EMOTION_FILES[meta.language][meta.emotion][meta.gender].append(
                        {
                            "path": os.path.join(root, file),
                            "gender": meta.gender,
                            "id": meta.id,
                        }
                    )

                    if meta.id > MAX_IDS[meta.language][meta.emotion]:
                        MAX_IDS[meta.language][meta.emotion] = meta.id
                except ValueError as e:
                    print(f"Error {file}: {e}")
                    continue


def balance_all_emotions():
    """
    Balances the number of audio files across all emotions and gender by duplicating existing files.
    For each language-emotion combination that has fewer files than the maximum count,
    this function creates copies of existing files to reach the maximum count.
    """
    max_val = max(
        count
        for language_dict in AUDIO_DICT.values()
        for count in language_dict.values()
    )

    for language in AUDIO_DICT.keys():
        for emotion in AUDIO_DICT[language].keys():
            current_count = AUDIO_DICT[language][emotion]
            if current_count < max_val:
                needed = max_val - current_count

                gender_files = EMOTION_FILES[language][emotion]

                available_genders = list(gender_files.keys())

                for i in range(needed):
                    if REVERSE_ORDER[language][emotion]:
                        current_gender = available_genders[i % len(available_genders)]
                        if len(available_genders) > 1 and i % 2 == 0:
                            current_gender = (
                                available_genders[1]
                                if available_genders[0] == "F"
                                else available_genders[0]
                            )
                    else:
                        current_gender = available_genders[i % len(available_genders)]
                        if len(available_genders) > 1 and i % 2 == 1:
                            current_gender = (
                                available_genders[1]
                                if available_genders[0] == "F"
                                else available_genders[0]
                            )

                    if current_gender in gender_files and gender_files[current_gender]:
                        file_idx = DUPLICATION_INDICES[language][emotion][
                            current_gender
                        ] % len(gender_files[current_gender])
                        source_info = gender_files[current_gender][file_idx]
                        DUPLICATION_INDICES[language][emotion][current_gender] += 1

                        new_id = MAX_IDS[language][emotion] + i + 1
                        new_file_name = (
                            f"{language}_{source_info['gender']}_{emotion}_{new_id}.wav"
                        )
                        new_file_path = os.path.join(
                            os.path.dirname(source_info["path"]), new_file_name
                        )
                        shutil.copy2(source_info["path"], new_file_path)

                REVERSE_ORDER[language][emotion] = not REVERSE_ORDER[language][emotion]

    audio_dict_updated = defaultdict(lambda: defaultdict(int))
    for root, directories, files in os.walk("src//data//dataset"):
        for file in files:
            if file.lower().endswith(".wav"):
                try:
                    meta = get_audio_metadado(file)
                    audio_dict_updated[meta.language][meta.emotion] += 1

                except ValueError as e:
                    print(f"Error {file}: {e}")
                    continue

    return audio_dict_updated


if __name__ == "__main__":
    read = read_data()
    balance = balance_all_emotions()
