"""Classification model module."""
import mlflow
from typing import Tuple, Dict
from typing_extensions import Self

import tensorflow as tf
from keras import layers, Sequential, losses, Model
from keras.src.callbacks import History
from keras.models import load_model

from server.dataset import load_tf_dataset
from server.dvc import DVCFileVersion

MODEL_PATH = "models/classifier.keras"


def get_untrained_model() -> Model:
    """Get a basic sequential model to be fitted.

    Returns:
        Sequential: model to be fitted
    """
    num_classes = 2

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(128, 128, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print(model.summary())

    return model


class TerminatorClassifier:
    """Terminator Classifier."""
    def __init__(self, model: Model) -> None:
        """Initialise Terminator classifier.

        Args:
            model (Model): tensorflow model for classification
        """
        self.model = model

    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> History:
        """Fit a sequential model.

        Args:
            train_dataset (tf.data.Dataset): train set
            val_dataset (tf.data.Dataset): validation set
            epochs (int): number of epochs to fit for

        Returns:
            history of the fitting
        """
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs
        )

        return history

    def evaluate(self, test_dataset: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate model performance.

        Args:
            test_dataset (tf.data.Dataset): test dataset

        Returns:
            Tuple[float, float]: test loss and accuracy
        """
        return self.model.evaluate(test_dataset)

    @classmethod
    def get_untrained(cls) -> Self:
        """Get untrained Terminator Classifier.

        Returns:
            TerminatorClassifier: untrained classifier
        """
        return cls(get_untrained_model())

    @classmethod
    def from_path(cls, path: str) -> Self:
        """Initialise model from a previously trained one.

        Args:
            path (str): path to the saved model

        Return:
            TerminatorClassifier: trained classifier
        """
        return cls(load_model(path))

    def save(self, path: str) -> None:
        """Save model to the filesystem.

        Args:
            path (str): path to save to
        """
        self.model.save(path)


def run_model_training_experiment() -> None:
    """Run end to end experiment and log everything to MLFlow."""
    with mlflow.start_run() as run:
        mlflow.autolog()

        # Log dataset versions
        train_version = DVCFileVersion.from_filepath("data/train")
        mlflow.log_param("train_git_revisions", train_version.git_revisions)
        mlflow.log_param("train_committed_datetime", train_version.committed_datetime)

        test_version = DVCFileVersion.from_filepath("data/test")
        mlflow.log_param("test_git_revisions", test_version.git_revisions)
        mlflow.log_param("test_committed_datetime", test_version.committed_datetime)

        train_dataset, val_dataset = load_tf_dataset("train", split=True)
        test_dataset = load_tf_dataset("test", split=False)
        classifier = TerminatorClassifier.get_untrained()
        history = classifier.train(train_dataset, val_dataset, epochs=10)
        test_loss, test_accuracy = classifier.evaluate(test_dataset)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        classifier.save(MODEL_PATH)


def get_model_evaluation() -> Dict[str, float]:
    """Evaluate previously trained and saved model.

    Returns:
        dict: containing evaluation metrics
    """
    classifier = TerminatorClassifier.from_path(MODEL_PATH)
    test_dataset = load_tf_dataset("test", split=False)
    test_loss, test_accuracy = classifier.evaluate(test_dataset)
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

