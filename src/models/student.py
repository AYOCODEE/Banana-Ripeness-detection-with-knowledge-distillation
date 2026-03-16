"""Student model definition - custom lightweight ResNet10."""

import tensorflow as tf
from tensorflow.keras import layers, models, Input


def basic_block(x, filters, stride=1, downsample=False):
    """Create a basic residual block.

    Args:
        x: Input tensor.
        filters (int): Number of convolutional filters.
        stride (int): Stride for the first convolution. Defaults to 1.
        downsample (bool): Whether to apply downsampling to the identity.
            Defaults to False.

    Returns:
        Tensor: Output tensor after the residual block.
    """
    identity = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample:
        identity = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False)(identity)
        identity = layers.BatchNormalization()(identity)

    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    return x


def build_student_model(input_shape=(224, 224, 3), num_classes=4):
    """Build and return the custom lightweight ResNet10 student model.

    This model is designed to be trained via knowledge distillation from the
    teacher (ResNet50) model. It has significantly fewer parameters while
    maintaining reasonable accuracy.

    Args:
        input_shape (tuple): Input image shape (height, width, channels).
            Defaults to (224, 224, 3).
        num_classes (int): Number of output classes. Defaults to 4.

    Returns:
        keras.Model: Student model.

    Example:
        >>> model = build_student_model()
        >>> model.summary()
    """
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Block 1
    x = basic_block(x, 64)

    # Block 2
    x = basic_block(x, 128, stride=2, downsample=True)

    # Block 3
    x = basic_block(x, 192, stride=2, downsample=True)

    # Block 4
    x = basic_block(x, 256, stride=2, downsample=True)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs, outputs)
    return model


def compile_student_model(model, learning_rate=1e-3):
    """Compile the student model with Adam optimizer and sparse CE loss.

    Args:
        model (keras.Model): The student model to compile.
        learning_rate (float): Learning rate for the Adam optimizer.
            Defaults to 1e-3.

    Returns:
        keras.Model: Compiled student model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
