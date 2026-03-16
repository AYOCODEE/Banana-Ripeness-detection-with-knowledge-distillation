"""Unit tests for the inference pipeline."""

import os
import tempfile

import numpy as np
import pytest


def test_preprocess_image_shape(tmp_path):
    """preprocess_image returns array with shape (1, 224, 224, 3)."""
    from PIL import Image
    from src.inference import preprocess_image

    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    img_path = str(tmp_path / "test.jpg")
    img.save(img_path)

    result = preprocess_image(img_path)
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_image_normalised(tmp_path):
    """preprocess_image normalises pixel values to [0, 1]."""
    from PIL import Image
    from src.inference import preprocess_image

    img = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    img_path = str(tmp_path / "white.jpg")
    img.save(img_path)

    result = preprocess_image(img_path)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_image_missing_file():
    """preprocess_image raises FileNotFoundError for missing files."""
    from src.inference import preprocess_image

    with pytest.raises(FileNotFoundError):
        preprocess_image("/nonexistent/path/image.jpg")


def test_load_model_missing_path():
    """load_model raises FileNotFoundError for missing model."""
    from src.inference import load_model

    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/model/path")
