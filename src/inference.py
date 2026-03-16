"""Inference pipeline for Banana Ripeness Detection.

This module provides utilities to load a saved model and run predictions on
single images or batches of images from the command line.

Usage::

    python -m src.inference --model path/to/model --image path/to/image.jpg
"""

import argparse
import os

import numpy as np
from PIL import Image


CLASS_NAMES = ["Overripe", "Ripe", "Unripe", "Rotten"]
DEFAULT_IMG_SIZE = (224, 224)


def load_model(model_path):
    """Load a saved Keras model from disk.

    Args:
        model_path (str): Path to a saved Keras model directory or ``.h5``
            file.

    Returns:
        keras.Model: The loaded model.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.

    Example:
        >>> model = load_model("models/resnet10_distilled_student_model")
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path, img_size=DEFAULT_IMG_SIZE):
    """Load and preprocess a single image for inference.

    Resizes the image, converts to RGB, normalises pixel values to [0, 1]
    and adds a batch dimension.

    Args:
        image_path (str): Path to the input image file.
        img_size (tuple): Target size as (height, width). Defaults to
            (224, 224).

    Returns:
        np.ndarray: Preprocessed image array with shape (1, H, W, 3).

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.

    Example:
        >>> img = preprocess_image("banana.jpg")
        >>> print(img.shape)
        (1, 224, 224, 3)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model, image_array, class_names=CLASS_NAMES):
    """Run inference on a preprocessed image array.

    Args:
        model: A loaded Keras model with a ``predict`` method.
        image_array (np.ndarray): Preprocessed image array with shape
            (1, H, W, 3).
        class_names (list): Ordered list of class name strings.

    Returns:
        dict: A dictionary with keys:
            - ``predicted_class`` (str): The predicted class name.
            - ``confidence`` (float): Confidence score in [0, 1].
            - ``probabilities`` (dict): Per-class softmax probabilities.

    Example:
        >>> model = load_model("models/resnet10_distilled_student_model")
        >>> img = preprocess_image("banana.jpg")
        >>> result = predict(model, img)
        >>> print(result["predicted_class"], result["confidence"])
    """
    import tensorflow as tf

    logits = model.predict(image_array, verbose=0)
    probabilities = tf.nn.softmax(logits[0]).numpy()
    predicted_idx = int(np.argmax(probabilities))

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {
            name: float(prob) for name, prob in zip(class_names, probabilities)
        },
    }


def predict_from_path(model_path, image_path, class_names=CLASS_NAMES):
    """Convenience wrapper: load model, preprocess image, run prediction.

    Args:
        model_path (str): Path to a saved Keras model.
        image_path (str): Path to an input image file.
        class_names (list): Ordered list of class name strings.

    Returns:
        dict: Prediction result as returned by :func:`predict`.

    Example:
        >>> result = predict_from_path(
        ...     "models/resnet10_distilled_student_model",
        ...     "test_banana.jpg",
        ... )
        >>> print(result)
    """
    model = load_model(model_path)
    image_array = preprocess_image(image_path)
    return predict(model, image_array, class_names)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a single banana image."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = predict_from_path(args.model, args.image)
    print(f"Predicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence']:.2%}")
    print("Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<12}: {prob:.4f}")
