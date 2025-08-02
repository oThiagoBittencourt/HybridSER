import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from src.models.data_loader import get_data_loaders
from src.models.model import build_multimodal_model
from src.utils.utils import load_config
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore


def plot_history(history, model_name):

    """Plots training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{model_name} Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(f'{model_name}_training_history.png')
    plt.show()


def main():

    config = load_config()
    # 1. Load Data
    try:
        train_ds, val_ds, test_ds, label_encoder, zcr_scaler, rms_scaler = get_data_loaders(
            features_dir=config["FEATURES_DIR"],
            batch_size=config["BATCH_SIZE"]
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the 'features' directory exists and is populated correctly.")
        return

    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} emotion classes: {label_encoder.classes_}")

    img_shape = (config["IMG_HEIGHT"], config["IMG_WIDTH"], config["NUM_CHANNELS"])
    numerical_shape = (config["FIXED_1D_LENGTH"] * 2,)

    # 2. Build Model
    model = build_multimodal_model(
        img_shape=img_shape,
        numerical_shape=numerical_shape,
        num_classes=num_classes,
        zcr_scaler=zcr_scaler,
        rms_scaler=rms_scaler
    )

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # --- Define Callbacks ---
    # Save the best model based on validation loss
    model_checkpoint_callback = ModelCheckpoint(
        filepath='best_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    # Stop training if validation loss doesn't improve for 10 epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
    )

    # Reduce learning rate when a metric has stopped improving
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Create a list of callbacks to pass to model.fit()
    training_callbacks = [
        model_checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback
    ]

    # 4. Train Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["EPOCHS"],
        callbacks=training_callbacks  # Pass the list here
    )
    print("--- Model Training Finished ---\n")

    # 5. Evaluate Model
    print("--- Evaluating Model on Test Set ---")
    # Note: If EarlyStopping restored best weights, this evaluation uses the best model.
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 6. Visualize Results
    plot_history(history, model.name)


if __name__ == '__main__':
    main()
