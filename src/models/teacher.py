"""Teacher model definition using pretrained ResNet50."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout


def build_teacher_model(input_shape=(224, 224, 3), num_classes=4):
    """Build and return the teacher model based on pretrained ResNet50.

    The teacher model uses ResNet50 pretrained on ImageNet as a feature
    extractor, followed by custom classification layers.

    Args:
        input_shape (tuple): Input image shape (height, width, channels).
            Defaults to (224, 224, 3).
        num_classes (int): Number of output classes. Defaults to 4.

    Returns:
        keras.Model: Compiled teacher model.

    Example:
        >>> model = build_teacher_model()
        >>> model.summary()
    """
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs, outputs)
    return model


def compile_teacher_model(model, learning_rate=1e-3):
    """Compile the teacher model with Adam optimizer and sparse CE loss.

    Args:
        model (keras.Model): The teacher model to compile.
        learning_rate (float): Learning rate for the Adam optimizer.
            Defaults to 1e-3.

    Returns:
        keras.Model: Compiled teacher model.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
