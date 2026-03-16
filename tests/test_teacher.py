"""Unit tests for the teacher model."""

import pytest


@pytest.mark.requires_tf
def test_build_teacher_model_output_shape():
    """Teacher model output shape matches num_classes."""
    pytest.importorskip("tensorflow")
    from src.models.teacher import build_teacher_model

    model = build_teacher_model(input_shape=(224, 224, 3), num_classes=4)
    assert model.output_shape == (None, 4)


@pytest.mark.requires_tf
def test_build_teacher_model_custom_classes():
    """Teacher model respects custom num_classes."""
    pytest.importorskip("tensorflow")
    from src.models.teacher import build_teacher_model

    model = build_teacher_model(input_shape=(224, 224, 3), num_classes=3)
    assert model.output_shape == (None, 3)


@pytest.mark.requires_tf
def test_compile_teacher_model():
    """compile_teacher_model returns a compiled model."""
    pytest.importorskip("tensorflow")
    from src.models.teacher import build_teacher_model, compile_teacher_model

    model = build_teacher_model(input_shape=(224, 224, 3), num_classes=4)
    compiled = compile_teacher_model(model)
    assert compiled.optimizer is not None
