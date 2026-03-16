"""Visualisation utilities for training metrics and predictions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """Plot training and validation accuracy and loss curves.

    Args:
        history: A Keras ``History`` object returned by ``model.fit()``.
        save_path (str, optional): If provided, the figure is saved to this
            path instead of being displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> history = model.fit(train_ds, ...)
        >>> plot_training_history(history, save_path="results/history.png")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Determine accuracy key
    acc_key = (
        "sparse_categorical_accuracy"
        if "sparse_categorical_accuracy" in history.history
        else "accuracy"
    )
    val_acc_key = f"val_{acc_key}"

    axes[0].plot(history.history[acc_key], label="Training Accuracy")
    axes[0].plot(history.history[val_acc_key], label="Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_distiller_history(history, save_path=None):
    """Plot training metrics from a Distiller training run.

    Args:
        history: A Keras ``History`` object returned by
            ``distiller.fit()``.
        save_path (str, optional): If provided, the figure is saved to this
            path instead of being displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> history = distiller.fit(train_ds, ...)
        >>> plot_distiller_history(history, save_path="results/distiller.png")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].plot(history.history["sparse_categorical_accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_sparse_categorical_accuracy"], label="Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Distiller Training and Validation Accuracy")

    axes[1].plot(history.history["student_loss"], label="Training Loss")
    axes[1].plot(history.history["val_student_loss"], label="Validation Loss")
    axes[1].plot(history.history["distillation_loss"], label="Distillation Loss")
    axes[1].plot(history.history["loss"], label="Total Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].set_title("Distiller Training and Validation Loss")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


def plot_sample_predictions(images, true_labels, pred_labels, class_names, n=9, save_path=None):
    """Display a grid of sample images with their true and predicted labels.

    Args:
        images (np.ndarray): Array of images with shape (N, H, W, C).
        true_labels (np.ndarray): True class indices.
        pred_labels (np.ndarray): Predicted class indices.
        class_names (list): List of class name strings.
        n (int): Number of samples to display (must be a perfect square).
            Defaults to 9.
        save_path (str, optional): If provided, save figure to this path.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i]
        # Normalise to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        ax.imshow(img)
        true_name = class_names[int(true_labels[i])]
        pred_name = class_names[int(pred_labels[i])]
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        ax.set_title(f"T:{true_name}\nP:{pred_name}", color=color, fontsize=8)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig
