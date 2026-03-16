"""Data loading and preprocessing utilities for Banana Ripeness Detection."""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Default class names matching the original dataset structure
CLASS_NAMES = ["Overripe", "Ripe", "Unripe", "Rotten"]


def create_data_generators(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.3,
    seed=42,
):
    """Create training, validation and test data generators.

    Applies data augmentation (rotation, zoom, horizontal flip) only to the
    training split. Validation and test splits are un-augmented.

    Args:
        data_dir (str): Path to the root dataset directory.  The directory
            must follow the ``flow_from_directory`` format, i.e. one
            sub-folder per class.
        img_size (tuple): Target image size as (height, width).
            Defaults to (224, 224).
        batch_size (int): Number of images per batch. Defaults to 32.
        validation_split (float): Fraction of images reserved for
            validation/testing. Defaults to 0.3.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A 3-tuple of
            (train_generator, val_generator, test_generator).

    Raises:
        ValueError: If ``data_dir`` does not exist.

    Example:
        >>> train_gen, val_gen, test_gen = create_data_generators("data/raw")
        >>> print(train_gen.samples)
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir does not exist: {data_dir}")

    train_datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    val_test_datagen = ImageDataGenerator(
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=seed,
    )

    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=True,
        seed=seed,
    )

    test_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    return train_generator, val_generator, test_generator


def get_dataset_info(generator):
    """Return a summary dict of a data generator.

    Args:
        generator: A Keras ``DirectoryIterator`` (from
            ``flow_from_directory``).

    Returns:
        dict: Dictionary with keys ``num_samples``, ``num_classes``,
            ``class_indices``, and ``batch_size``.

    Example:
        >>> info = get_dataset_info(train_gen)
        >>> print(info["num_samples"])
    """
    return {
        "num_samples": generator.samples,
        "num_classes": generator.num_classes,
        "class_indices": generator.class_indices,
        "batch_size": generator.batch_size,
    }
