"""Unit tests for metrics utilities."""

import pytest
from src.utils.metrics import compute_accuracy, get_class_report


def test_compute_accuracy_perfect():
    """Perfect predictions give accuracy 1.0."""
    assert compute_accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_compute_accuracy_zero():
    """Completely wrong predictions give accuracy 0.0."""
    assert compute_accuracy([0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)


def test_compute_accuracy_partial():
    """Partial accuracy computed correctly."""
    assert compute_accuracy([0, 1, 2, 3], [0, 1, 0, 0]) == pytest.approx(0.5)


def test_get_class_report_keys():
    """Report contains all class names."""
    class_names = ["Overripe", "Ripe", "Unripe", "Rotten"]
    report = get_class_report([0, 1, 2, 3], [0, 1, 2, 3], class_names)
    assert set(report.keys()) == set(class_names)


def test_get_class_report_perfect_f1():
    """Perfect predictions give F1 = 1.0 for all classes."""
    class_names = ["A", "B"]
    report = get_class_report([0, 1, 0, 1], [0, 1, 0, 1], class_names)
    for name in class_names:
        assert report[name]["f1"] == pytest.approx(1.0)


def test_get_class_report_support():
    """Support count matches number of true samples per class."""
    class_names = ["A", "B", "C"]
    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]
    report = get_class_report(y_true, y_pred, class_names)
    assert report["A"]["support"] == 2
    assert report["B"]["support"] == 1
    assert report["C"]["support"] == 3
