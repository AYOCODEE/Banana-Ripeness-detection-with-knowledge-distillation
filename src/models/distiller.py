"""Knowledge Distillation training logic."""

import tensorflow as tf
from tensorflow import keras


class Distiller(keras.Model):
    """Knowledge distillation model that trains a student using a teacher.

    This class implements the knowledge distillation training procedure
    where a smaller student model learns from a larger, pre-trained teacher
    model by minimising a combination of the task loss and the distillation
    loss (KL-divergence between softened teacher/student logits).

    Args:
        student (keras.Model): The student model to train.
        teacher (keras.Model): The pre-trained teacher model (frozen).

    Example:
        >>> distiller = Distiller(student=student_model, teacher=teacher_model)
        >>> distiller.compile(
        ...     optimizer=keras.optimizers.Adam(),
        ...     metrics=[keras.metrics.SparseCategoricalAccuracy()],
        ...     student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ...     distillation_loss_fn=keras.losses.KLDivergence(),
        ...     alpha=0.1,
        ...     temperature=40,
        ... )
        >>> distiller.fit(train_dataset, epochs=10, validation_data=val_dataset)
    """

    def __init__(self, student, teacher):
        """Initialise the Distiller with student and teacher models.

        Args:
            student (keras.Model): The student model to train.
            teacher (keras.Model): The pre-trained teacher model (frozen).
        """
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights.
            metrics: Keras metrics for evaluation.
            student_loss_fn: Loss function between student predictions and
                ground-truth labels.
            distillation_loss_fn: Loss function between soft student
                predictions and soft teacher predictions.
            alpha (float): Weight for the student task loss. The distillation
                loss is weighted by (1 - alpha). Defaults to 0.1.
            temperature (float): Temperature for softening probability
                distributions. Larger values produce softer distributions.
                Defaults to 3.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        """Perform a single training step.

        Args:
            data: A tuple (x, y) of input images and labels.

        Returns:
            dict: Metric results including student_loss and distillation_loss.
        """
        x, y = data

        # Forward pass of teacher (no gradient tracking needed)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute task loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute distillation loss on softened logits
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute and apply gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        """Perform a single evaluation step.

        Args:
            data: A tuple (x, y) of input images and labels.

        Returns:
            dict: Metric results including student_loss.
        """
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update metrics
        self.compiled_metrics.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
