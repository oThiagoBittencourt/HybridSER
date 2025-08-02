import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import ResNet50  # type: ignore


def create_visual_branch(input_tensor, name):
    """
    Creates a visual processing branch using a pre-trained ResNet50.
    This function now accepts a Keras tensor as input and uses its shape.
    """
    # Get the shape from the input tensor, excluding the batch size
    input_shape = input_tensor.shape[1:]

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        name=f"resnet50_{name}"  # This creates "resnet50_mfcc" and "resnet50_chroma"
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Pass the input tensor through the base model
    x = base_model(input_tensor, training=False)

    # Add a global pooling layer
    x = GlobalAveragePooling2D(name=f"gap_{name}")(x)

    # Return the final output tensor of this branch
    return x


def create_numerical_branch(numerical_input, zcr_scaler, rms_scaler):
    """
    Builds the branch of the model that processes numerical features.
    """
    # Split the 1024-feature input back into its original 512-feature parts
    zcr_features = numerical_input[:, :512]  # First 512 features
    rms_features = numerical_input[:, 512:]  # Last 512 features

    # --- Apply Scaling Separately ---
    # Convert scaler attributes to TensorFlow constants
    zcr_mean = tf.constant(zcr_scaler.mean_, dtype=tf.float32)
    zcr_scale = tf.constant(zcr_scaler.scale_, dtype=tf.float32)
    zcr_scaled = (zcr_features - zcr_mean) / zcr_scale

    rms_mean = tf.constant(rms_scaler.mean_, dtype=tf.float32)
    rms_scale = tf.constant(rms_scaler.scale_, dtype=tf.float32)
    rms_scaled = (rms_features - rms_mean) / rms_scale

    # Concatenate the individually scaled features back together
    concatenated_scaled = layers.Concatenate()([zcr_scaled, rms_scaled])

    x = layers.Dense(512, activation='relu')(concatenated_scaled)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)

    return x


def build_multimodal_model(img_shape, numerical_shape, num_classes, zcr_scaler, rms_scaler):
    """
    Builds the complete multi-input model.
    """
    # Define the input layers for the image branches
    mfcc_input = layers.Input(shape=img_shape, name="mfcc_input")
    chroma_input = layers.Input(shape=img_shape, name="chroma_input")
    numerical_input = layers.Input(shape=numerical_shape, name="numerical_input")

    # Build the individual branches by passing the Input TENSORS and a name
    mfcc_branch = create_visual_branch(mfcc_input, name="mfcc")
    chroma_branch = create_visual_branch(chroma_input, name="chroma")
    numerical_branch = create_numerical_branch(numerical_input, zcr_scaler, rms_scaler)

    # Concatenate the outputs of all branches
    combined_features = layers.Concatenate()([mfcc_branch, chroma_branch, numerical_branch])

    # Add the final classification head
    x = layers.Dense(128, activation='relu')(combined_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(
        inputs=[mfcc_input, chroma_input, numerical_input],
        outputs=output
    )

    return model
