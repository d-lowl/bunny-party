import os.path
from unittest import mock

from server.dataset import relabel_image_in_dataset, load_tf_dataset, load_class_names, load_images


def test_load_dataset():
    with mock.patch("tensorflow.keras.utils.image_dataset_from_directory") as mock_dataset_loader:
        load_tf_dataset("dataset_name", split=False)
        mock_dataset_loader.assert_called_with("data/dataset_name", image_size=(128, 128))


def test_load_split_dataset():
    with mock.patch("tensorflow.keras.utils.image_dataset_from_directory") as mock_dataset_loader:
        load_tf_dataset("dataset_name", split=True)
        mock_dataset_loader.assert_called_with(
            "data/dataset_name",
            image_size=(128, 128),
            validation_split=0.2,
            subset="both",
            seed=101,
        )


def test_relabel_image_in_dataset():
    with mock.patch("os.rename") as mock_rename:
        relabel_image_in_dataset(
            "test_dataset",
            "image0",
            "1",
            "0"
        )
        mock_rename.assert_called_with(
            "data/test_dataset/1/image0.jpg",
            "data/test_dataset/0/image0.jpg"
        )


def test_load_class_names():
    """Test loading class names."""
    path = "data/class_name_to_id_mapping.txt"  # assumed
    with (mock.patch("builtins.open") as mock_open, mock.patch("json.load") as mock_json_load):
        expected_class_names = {
            "0": "enemy",
            "1": "friend"
        }
        mock_json_load.return_value = expected_class_names
        class_names = load_class_names()
        assert expected_class_names == class_names
        mock_open.assert_called_with(path)


def test_load_images():
    """Test loading image collection."""
    dataset_name = "test_dataset"
    dataset_path = os.path.join("data", dataset_name)
    expected_patterns = [
        "data/test_dataset/0/*.jpg",
        "data/test_dataset/1/*.jpg",
    ]
    with (
        mock.patch("server.dataset.io.ImageCollection") as mock_image_collection,
        mock.patch("os.listdir", return_value=["0", "1"]) as mock_listdir
    ):
        collection = load_images(dataset_name)
        assert collection == mock_image_collection.return_value
        mock_image_collection.assert_called_with(expected_patterns)
        mock_listdir.assert_called_with(dataset_path)

