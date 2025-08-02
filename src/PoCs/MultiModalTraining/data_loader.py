import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.utils import load_config
from PIL import Image  # Required for image conversion


def convert_png_to_jpeg(directory):
    """
    Recursively finds all .png files in a directory, converts them to .jpeg,
    and removes the original .png file.
    """
    print("Starting PNG to JPEG conversion...")
    converted_count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.png'):
                png_path = os.path.join(root, filename)
                # Create the new jpeg path with the same name
                jpeg_path = os.path.splitext(png_path)[0] + '.jpeg'

                try:
                    with Image.open(png_path) as img:
                        # Convert to RGB if it has an alpha channel (like RGBA)
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        img.save(jpeg_path, 'jpeg', quality=90)  # quality can be adjusted

                    # Remove the original PNG file after successful conversion
                    os.remove(png_path)
                    converted_count += 1
                    print(f"Converted and removed: {png_path}")

                except Exception as e:
                    print(f"Could not convert {png_path}: {e}")

    print(f"\nConversion complete. Total files converted: {converted_count}")


def parse_filepaths(features_dir):
    """
    Walks the features directory to parse file paths and extract labels.
    Now looks for .jpeg files instead of .png.
    """
    filepaths = []
    total_dirs_scanned = 0
    dirs_failed_name_check = 0
    dirs_failed_file_check = 0

    print(f"Starting scan of: {features_dir}")

    for root, _, files in os.walk(features_dir):
        if not files:
            continue

        total_dirs_scanned += 1
        dir_name = os.path.basename(root)
        parts = dir_name.split("_")
        lang, gender, emotion, index = None, None, None, None

        if len(parts) == 3:
            lang, gender, emotion = parts
            index = "0"
        elif len(parts) == 4 and parts[3].isdigit():
            lang, gender, emotion = parts[0], parts[1], parts[2]
            index = parts[3]
        else:
            dirs_failed_name_check += 1
            continue

        current_sample = {"language": lang, "gender": gender, "emotion": emotion, "index": int(index)}
        for f in files:
            # --- MODIFIED TO FIND .jpeg ---
            if "dd_mfcc.jpeg" in f:
                current_sample["mfcc_path"] = os.path.join(root, f)
            elif "chromagram.jpeg" in f:
                current_sample["chromagram_path"] = os.path.join(root, f)
            elif "zcr.npy" in f:
                current_sample["zcr_path"] = os.path.join(root, f)
            elif "rms.npy" in f:
                current_sample["rms_path"] = os.path.join(root, f)

        required_keys = ["mfcc_path", "chromagram_path", "zcr_path", "rms_path"]
        if all(key in current_sample for key in required_keys):
            filepaths.append(current_sample)
        else:
            dirs_failed_file_check += 1

    print("\n--- Parsing Summary ---")
    print(f"Total directories with files scanned: {total_dirs_scanned}")
    print(f"Directories that failed name check: {dirs_failed_name_check}")
    print(f"Directories that failed file check (missing files): {dirs_failed_file_check}")
    print(f"Successfully parsed directories: {len(filepaths)}")
    print("--------------------------\n")

    return pd.DataFrame(filepaths)


def load_and_preprocess(mfcc_path, chromagram_path, zcr_path, rms_path, label):
    """
    Loads and preprocesses a single data sample for the multi-input model.
    """
    config = load_config()

    def _load_image(path):
        img_raw = tf.io.read_file(path)
        # --- MODIFIED TO DECODE .jpeg ---
        img = tf.image.decode_jpeg(img_raw, channels=config["NUM_CHANNELS"])
        img = tf.image.resize(img, [config["IMG_HEIGHT"], config["IMG_WIDTH"]])
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    mfcc_img = _load_image(mfcc_path)
    chroma_img = _load_image(chromagram_path)

    def _load_npy(path):
        data = np.load(path.numpy().decode("utf-8"))
        if len(data) > config["FIXED_1D_LENGTH"]:
            data = data[:config["FIXED_1D_LENGTH"]]
        else:
            data = np.pad(data, (0, config["FIXED_1D_LENGTH"] - len(data)), "constant")
        return data.astype(np.float32)

    zcr_data = tf.py_function(_load_npy, [zcr_path], tf.float32)
    rms_data = tf.py_function(_load_npy, [rms_path], tf.float32)
    zcr_data.set_shape((config["FIXED_1D_LENGTH"],))
    rms_data.set_shape((config["FIXED_1D_LENGTH"],))
    numerical_features = tf.concat([zcr_data, rms_data], axis=0)

    inputs = {
        "mfcc_input": mfcc_img,
        "chroma_input": chroma_img,
        "numerical_input": numerical_features,
    }
    return inputs, label


def create_dataset(df, label_encoder, zcr_scaler, rms_scaler, batch_size):
    """
    Creates a tf.data.Dataset from a pandas DataFrame.
    """
    df["emotion_encoded"] = label_encoder.transform(df["emotion"])
    labels_one_hot = tf.keras.utils.to_categorical(
        df["emotion_encoded"], num_classes=len(label_encoder.classes_)
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["mfcc_path"].values,
            df["chromagram_path"].values,
            df["zcr_path"].values,
            df["rms_path"].values,
            labels_one_hot,
        )
    )

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_data_loaders(features_dir, batch_size=32, test_size=0.2, val_size=0.2):
    """
    Main function to parse data, create splits, and return tf.data.Dataset objects.
    """
    config = load_config()
    df = parse_filepaths(features_dir)
    if df.empty:
        raise ValueError("No feature files found or parsed.")

    print("--- Cleaning Labels ---")
    print("\nBefore cleaning:")
    print(df['emotion'].value_counts())
    label_map = {
        'Angry': 'Anger', 'Sad': 'Sadness', 'Surprised': 'Surprise', 'Calm' : 'Neutral'
    }
    df['emotion'] = df['emotion'].replace(label_map).str.capitalize()
    print("\nAfter cleaning:")
    print(df['emotion'].value_counts())
    print("-------------------------\n")

    label_encoder = LabelEncoder()
    df["emotion_encoded"] = label_encoder.fit_transform(df["emotion"])

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["emotion_encoded"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_df["emotion_encoded"],
    )

    print("Fitting scalers on training data...")
    zcr_train_data = np.vstack(
        [
            (
                np.load(p)[:config["FIXED_1D_LENGTH"]]
                if len(np.load(p)) > config["FIXED_1D_LENGTH"]
                else np.pad(np.load(p), (0, config["FIXED_1D_LENGTH"] - len(np.load(p))))
            )
            for p in train_df["zcr_path"]
        ]
    )
    rms_train_data = np.vstack(
        [
            (
                np.load(p)[:config["FIXED_1D_LENGTH"]]
                if len(np.load(p)) > config["FIXED_1D_LENGTH"]
                else np.pad(np.load(p), (0, config["FIXED_1D_LENGTH"] - len(np.load(p))))
            )
            for p in train_df["rms_path"]
        ]
    )
    zcr_scaler = StandardScaler().fit(zcr_train_data)
    rms_scaler = StandardScaler().fit(rms_train_data)
    print("Scalers fitted.")

    print("Creating TensorFlow datasets...")
    train_ds = create_dataset(train_df, label_encoder, zcr_scaler, rms_scaler, batch_size)
    val_ds = create_dataset(val_df, label_encoder, zcr_scaler, rms_scaler, batch_size)
    test_ds = create_dataset(test_df, label_encoder, zcr_scaler, rms_scaler, batch_size)
    print("Datasets created.")

    return train_ds, val_ds, test_ds, label_encoder, zcr_scaler, rms_scaler


# --- Main execution block for one-time conversion ---
# To run, use the command: python -m src.models.data_loader
# IMPORTANT: Comment out or remove this block after the conversion is complete.
# if __name__ == '__main__':
#    config = load_config()
#    features_directory = config["FEATURES_DIR"]
#    print(f"Starting conversion in directory: {features_directory}")
#    convert_png_to_jpeg(features_directory)
#    print("Conversion complete.")
