from unittest import mock

import pytest
import tensorflow_datasets as tfds

from server.model import get_untrained_model, TerminatorClassifier


@pytest.fixture
def mock_dataset() -> mock.MagicMock:
    """Mock TF dataset."""
    with mock.patch("tensorflow.data.Dataset") as mock_dataset_class:
        mock_dataset = mock_dataset_class.return_value
        yield mock_dataset


@pytest.fixture
def mock_tf_model() -> mock.MagicMock:
    """Mock TF model instance."""
    with mock.patch("server.model.Sequential") as model_constructor:
        yield model_constructor.return_value


def test_get_untrained_model():
    """Test that a Sequential model is constructed and compiled"""
    with mock.patch("server.model.Sequential") as mock_sequential:
        returned_model = get_untrained_model()
        mock_sequential.assert_called()
        mock_model = mock_sequential.return_value
        mock_model.compile.assert_called()
        assert mock_model == returned_model


def test_terminator_classifier(mock_tf_model: mock.MagicMock):
    """Test the initialiser."""
    classifier = TerminatorClassifier(mock_tf_model)
    assert classifier.model == mock_tf_model


def test_terminator_classifier_train(
        mock_tf_model: mock.MagicMock,
        mock_dataset: mock.MagicMock
):
    """Test the training procedure."""
    classifier = TerminatorClassifier(mock_tf_model)
    with tfds.testing.mock_data(num_examples=5):
        classifier.train(
            mock_dataset,
            mock_dataset,
            5
        )
        mock_tf_model.fit.assert_called_with(
            mock_dataset,
            validation_data=mock_dataset,
            epochs=5
        )


def test_evaluate(
        mock_tf_model: mock.MagicMock,
        mock_dataset: mock.MagicMock
):
    """Test model evaluation method."""
    expected_metrics = (1.5, 0.5)  # Set mock loss and accuracy
    mock_tf_model.evaluate.return_value = expected_metrics
    classifier = TerminatorClassifier(mock_tf_model)
    got_metrics = classifier.evaluate(mock_dataset)
    assert expected_metrics == got_metrics
    mock_tf_model.evaluate.assert_called_with(mock_dataset)


def test_save(mock_tf_model: mock.MagicMock):
    """Test model saving method"""
    classifier = TerminatorClassifier(mock_tf_model)
    classifier.save("test/model/path.keras")
    mock_tf_model.save.assert_called_with("test/model/path.keras")


def test_from_path():
    """Test initialising Terminator Classifier with the previously saved model"""
    path = "test/model/path.keras"
    with mock.patch("server.model.load_model") as mock_load_model:
        classifier = TerminatorClassifier.from_path(path)
        assert classifier.model == mock_load_model.return_value
        mock_load_model.assert_called_with(path)