"""Evaluation metrics and reporting utilities."""

import numpy as np


def compute_accuracy(y_true, y_pred):
    """Compute top-1 classification accuracy.

    Args:
        y_true (array-like): True class indices, shape (N,).
        y_pred (array-like): Predicted class indices, shape (N,).

    Returns:
        float: Accuracy in the range [0, 1].

    Example:
        >>> acc = compute_accuracy([0, 1, 2], [0, 1, 0])
        >>> print(f"{acc:.2%}")
        66.67%
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def get_class_report(y_true, y_pred, class_names):
    """Return a per-class precision, recall and F1 report as a dict.

    This is a lightweight alternative to sklearn's classification_report
    that requires no additional dependencies.

    Args:
        y_true (array-like): True class indices, shape (N,).
        y_pred (array-like): Predicted class indices, shape (N,).
        class_names (list): Ordered list of class name strings.

    Returns:
        dict: Nested dict mapping class name ->
            {``precision``, ``recall``, ``f1``, ``support``}.

    Example:
        >>> report = get_class_report([0,1,1,2], [0,1,0,2], ["A","B","C"])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = {}
    for idx, name in enumerate(class_names):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return report


def print_class_report(report):
    """Pretty-print the output of :func:`get_class_report`.

    Args:
        report (dict): Report dict as returned by :func:`get_class_report`.
    """
    header = f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(header)
    print("-" * len(header))
    for class_name, metrics in report.items():
        print(
            f"{class_name:<20} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} "
            f"{metrics['support']:>10d}"
        )
