"""Unit tests for the student model."""

import pytest


@pytest.mark.requires_tf
def test_build_student_model_output_shape():
    """Student model output shape matches num_classes."""
    pytest.importorskip("tensorflow")
    from src.models.student import build_student_model

    model = build_student_model(input_shape=(64, 64, 3), num_classes=4)
    assert model.output_shape == (None, 4)


@pytest.mark.requires_tf
def test_build_student_model_custom_classes():
    """Student model respects custom num_classes."""
    pytest.importorskip("tensorflow")
    from src.models.student import build_student_model

    model = build_student_model(input_shape=(64, 64, 3), num_classes=3)
    assert model.output_shape == (None, 3)


@pytest.mark.requires_tf
def test_compile_student_model():
    """compile_student_model returns a compiled model."""
    pytest.importorskip("tensorflow")
    from src.models.student import build_student_model, compile_student_model

    model = build_student_model(input_shape=(64, 64, 3), num_classes=4)
    compiled = compile_student_model(model)
    assert compiled.optimizer is not None
