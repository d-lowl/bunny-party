from typing import Dict, List

import numpy as np
import pandas as pd
from skimage import io
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

from bunny_server.dataset import load_class_names, load_images, relabel_image_in_dataset


def get_vector_database(filepaths: List[str], embedded_images: np.ndarray, class_names: Dict[str, str]) -> pd.DataFrame:
    """Form a vector 'database' using the embeddings and metadata.

    Note: this implementation uses pandas, where's in a real scenario something
    like Chroma would be used directly

    Args:
        filepaths (str): paths to the original image files
        embedded_images (np.ndarray): 2D array of image embeddings
        class_names (Dict[str, str]): a mapping between class ids and names

    Returns:
        pd.DataFrame:
            Columns:
                id (str): image ID
                class_id (str): image's class id
                class_name (str): mapped image's class name
                embedding (np.ndarray): a vector to be used for similarity search
    """

    def get_row_from_filepath(filepath: str) -> pd.Series:
        """Infer class and image filename from its path.

        Returns:
            pd.Series:
                Columns:
                    id (str): image ID
                    class_id (str): image's class id
                    class_name (str): mapped image's class name
        """
        _, dataset_name, class_id, filename = filepath.split("/")
        class_name = class_names[class_id]
        return pd.Series({
            "id": filename.split(".")[0],
            "class_id": class_id,
            "class_name": class_name
        })

    df = pd.DataFrame([get_row_from_filepath(filepath) for filepath in filepaths])
    df["embedding"] = embedded_images.tolist()
    return df


class TerminatorEmbedding:
    """Terminator Embedding for similarity search."""

    _dataset_name: str
    _class_names: Dict[str, str]
    _collection: io.ImageCollection
    _embedding: PCA
    _embedded_images: np.ndarray
    dataframe: pd.DataFrame

    def __init__(self, dataset_name: str) -> None:
        """Initialise Terminator Embedding.

        Note: this embedding class is generated on the fly and stored in-memory due to time limitations
        of the exersice. In the real world, some persistent solution like ChromaDB would have been used.

        Args:
            dataset_name (str): name of the dataset to load
        """
        self._dataset_name = dataset_name
        self._class_names = load_class_names()
        self._collection = load_images(dataset_name)
        self._embedding = PCA(n_components=10)
        no_samples = len(self._collection)
        self._embedded_images = self._embedding.fit_transform(np.reshape(self._collection, (no_samples, -1)))
        self.dataframe = get_vector_database(self._collection.files, self._embedded_images, self._class_names)

    def get_distance(self, id: str) -> pd.DataFrame:
        """Get distance between the image given by ID and each element of the embedded dataset

        Args:
            id (str): image ID

        Returns:
            pd.DataFrame: copy of the dataframe, with `distance: float` column added to it.
        """
        distance = pd.DataFrame(euclidean_distances(
            np.stack(self.dataframe.embedding),
            np.stack(self.dataframe[self.dataframe.id == id].embedding)
        ), columns=["distance"], index=self.dataframe.index)

        return pd.concat([self.dataframe, distance], axis=1)

    def get_n_most_similar(self, img_id: str, n: int, class_id: str) -> pd.DataFrame:
        """Return N images that are most similar to a given image within a class.

        Args:
            img_id (str): image ID to compare with
            n (int): number of returned images
            class_id (str): class ID to search in

        Returns:
            pd.DataFrame: a slice of top N similar images
        """
        df = self.get_distance(img_id)
        return df[(df.id != img_id) & (df.class_id == class_id)].sort_values("distance").head(n)

    def relabel(self, img_id: str, class_id: str) -> None:
        """Relabel an image in-place within the embedding database.

        Args:
            img_id (str): image ID
            class_id (str): class ID
        """
        # Assuming the structure, there's only one entry that satisfies the condition
        old_class_id = self.dataframe[self.dataframe.id == img_id].class_id.iloc[0]
        self.dataframe.loc[self.dataframe.id == img_id, ["class_id"]] = class_id
        self.dataframe.loc[self.dataframe.id == img_id, ["class_name"]] = self._class_names[class_id]
        relabel_image_in_dataset(self._dataset_name, img_id, old_class_id, class_id)
