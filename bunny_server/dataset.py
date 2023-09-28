"""Dataset."""
import json
import os

from skimage import io
from typing import Tuple, Union, Dict

import tensorflow as tf


def load_class_names() -> Dict[str, str]:
    """Load class names.

    Returns:
        dict: a mapping between class ids and names
    """
    with open("data/class_name_to_id_mapping.txt") as f:
        return json.load(f)


def load_images(dataset_name: str) -> io.ImageCollection:
    """Load images as sklearn-image Image Collection

    Args:
        dataset_name (str): dataset name to load

    Returns:
        io.ImageCollection: image collection
    """
    patterns = [os.path.join("data", dataset_name, class_, "*.jpg") for class_ in
                os.listdir(os.path.join("data", dataset_name))]
    return io.ImageCollection(patterns)


def load_tf_dataset(
        name: str,
        split: bool = True
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """Load training dataset and split it into train and validation ones for Keras.

    Note: for the purpose of the exercise paths and sizes are not inferred.

    Returns:
        Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]: a dataset or a pair of train and val datasets
    """
    path = f"data/{name}"
    if split:
        return tf.keras.utils.image_dataset_from_directory(
            path,
            image_size=(128, 128),
            validation_split=0.2,
            subset="both",
            seed=101,
        )
    else:
        return tf.keras.utils.image_dataset_from_directory(
            path,
            image_size=(128, 128),
        )


def relabel_image_in_dataset(dataset_name: str, img_id: str, old_class: str, new_class: str) -> None:
    """Relabel an image by moving it into a different class directory.

    Args:
        dataset_name (str): dataset name
        img_id (str): image to relabel
        old_class (str): old class numeric label
        new_class (str): new class numeric label
    """
    old_path = os.path.join("data", dataset_name, old_class, f"{img_id}.jpg")
    new_path = os.path.join("data", dataset_name, new_class, f"{img_id}.jpg")
    os.rename(old_path, new_path)
