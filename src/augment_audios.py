import soundfile as sf
from pathlib import Path
from utils.augment import pitch_shift, time_stretch, add_awgn, tranform_estereo_to_mono

RAW_DIR = Path("src/audios/raw")
AUG_DIR = Path("src/audios/augmented")
AUG_DIR.mkdir(parents=True, exist_ok=True)
FILENAME = "ale-f1-l1.wav"

COMBINATIONS = [
    # "pitch",
    "time",
    # "awgn",
    # "pitch_time",
    # "pitch_awgn",
    # "time_pitch",
    # "time_awgn",
    "awgn_pitch",
    # "awgn_time",
    "pitch_time_awgn",
    # "pitch_awgn_time",
    # "time_pitch_awgn",
    # "time_awgn_pitch",
    # "awgn_pitch_time",
    # "awgn_time_pitch",
]


def apply_augmentations(y, sr, steps):
    for step in steps:
        if step == "pitch":
            y = pitch_shift(y, sr, steps=2)
        elif step == "time":
            y = time_stretch(y, rate=0.9)
        elif step == "awgn":
            y = add_awgn(y, snr_db=20)

        if len(y) < 2048:
            raise ValueError(f"A transformação '{step}' resultou em áudio muito curto (len={len(y)}). Abortando.")
    return y


def process_file(filepath, combinations):
    try:
        y, sr = sf.read(filepath)
        if y.ndim == 2:
            print(f"Áudio '{filepath.name}' é estéreo. Convertendo para mono.")
            y = tranform_estereo_to_mono(y)

        print(f"Carregando: {filepath.name} | sr={sr}, shape={y.shape}, duração ≈ {len(y)/sr:.2f}s")

        if len(y) < 2048:
            print(f"Áudio muito curto (len={len(y)}), ignorado.")
            return

        for combo in combinations:
            steps = combo.split("_")
            print(f"\nAplicando: '{combo}' em '{filepath.name}'...")
            y_aug = apply_augmentations(y, sr, steps)

            new_filename = f"{filepath.stem}_{combo}.wav"
            sf.write(AUG_DIR / new_filename, y_aug, sr)
            print(f"Arquivo gerado: {new_filename}")

    except Exception as e:
        print(f"Erro ao processar {filepath.name}: {e}")


def augment_all():
    files = list(RAW_DIR.glob("*.wav"))
    print(f"files: {files}")
    for file in files:
        process_file(file, COMBINATIONS)
        print("Todas as combinações foram aplicadas para todos os arquivos.")


def augment_single(filename):
    file = RAW_DIR / filename
    if not file.exists():
        print(f"Arquivo '{filename}' não encontrado em {RAW_DIR}")
        return
    process_file(file, COMBINATIONS)
    print(f"\nCombinações aplicadas para {filename}.")


if __name__ == "__main__":
    augment_single(FILENAME)
    # augment_all()
