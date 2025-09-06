from pathlib import Path
from .models.metadado import Metadado


def get_audio_metadado(file):
    """
    Extracts and parses metadata from audio filename to create a Metadado object.
    Returns a Metadado object containing language, gender, emotion, and ID information.
    """

    name_file = Path(file).stem
    parts = name_file.split("_")

    if len(parts) >= 4:
        metadado = Metadado(parts[0], parts[1], parts[2], int(parts[3]))

        return metadado
