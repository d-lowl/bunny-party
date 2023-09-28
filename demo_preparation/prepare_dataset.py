import hashlib
import json
import os
from typing import List

from PIL import Image
import random
from tqdm import tqdm

SIZE = (128, 128)

TRAIN_SAMPLE_NUMBER = 30
TEST_SAMPLE_NUMBER = 10

ORIGINAL_DATA_LOCATION = os.path.join(os.path.dirname(__file__), "data")
DATA_LOCATION = os.path.join(os.path.dirname(__file__), os.pardir, "data")


DOG = [
    "alvan-nee-brFsZ7qszSY-unsplash.jpg",
    "charlesdeluvio-Mv9hjnEUHR4-unsplash.jpg",
    "marcus-wallis-4zfacTKyZ7w-unsplash.jpg",
    "milli-2l0CWTpcChI-unsplash.jpg",
]

BUNNY = [
    "erik-jan-leusink---SDX4KWIbA-unsplash.jpg",
    "janan-lagerwall-302xfiIGOfE-unsplash.jpg",
    # "melanie-kreutz-IFnknR2Mv5o-unsplash.jpg",
    "satyabratasm-u_kMWN-BWyU-unsplash.jpg",
]

FAKE_BUNNY = "lovely-pet-portrait-isolated.jpg"

CLASSNAMES = {
    "0": "bunny",
    "1": "impostor"
}


def prepare(img: Image) -> Image:
    """Generate a random augmented image by rotation and resizing."""
    angle = random.uniform(0.0, 360.0)
    return img.rotate(angle, expand=True).resize(SIZE)


def get_name(img: Image) -> str:
    """Get md5 name for the new image."""
    return hashlib.md5(img.tobytes()).hexdigest()


def generate_dataset(
        dataset_name: str,
        class_label: str,
        image_names: List[str],
        samples_per_image: int,
) -> None:
    """Generate and write dataset.

    The dataset is written under: data/{dataset_name}/{class_label}

    Args:
        dataset_name (str): Name of the dataset (e.g. train, test).
        class_label (str): Numeric class label, see CLASSNAMES
        image_names (List[str]): List of the original images to generate samples
        samples_per_image (int): number of samples to generate from a given image
    """
    if not os.path.exists(os.path.join(DATA_LOCATION, dataset_name)):
        os.mkdir(os.path.join(DATA_LOCATION, dataset_name))
    if not os.path.exists(os.path.join(DATA_LOCATION, dataset_name, class_label)):
        os.mkdir(os.path.join(DATA_LOCATION, dataset_name, class_label))
    for image_name in tqdm(image_names):
        original = Image.open(os.path.join(ORIGINAL_DATA_LOCATION, "original", image_name))
        for _ in tqdm(range(samples_per_image)):
            prepared = prepare(original)
            img_name = get_name(prepared)
            prepared.save(os.path.join(DATA_LOCATION, dataset_name, class_label, f"{img_name}.jpg"))


if __name__ == "__main__":
    with open(os.path.join(DATA_LOCATION, "class_name_to_id_mapping.txt"), "w") as f:
        json.dump(CLASSNAMES, f, indent=4)

    random.seed(10)

    generate_dataset("train", "0", BUNNY + [FAKE_BUNNY], TRAIN_SAMPLE_NUMBER)
    generate_dataset("train", "1", DOG, TRAIN_SAMPLE_NUMBER)

    generate_dataset("test", "0", BUNNY, TEST_SAMPLE_NUMBER)
    generate_dataset("test", "1", DOG + [FAKE_BUNNY], TEST_SAMPLE_NUMBER)
