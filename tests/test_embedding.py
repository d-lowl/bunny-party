from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bunny_server.embedding import get_vector_database, TerminatorEmbedding


@pytest.fixture
def dummy_file_paths() -> List[str]:
    """Dummy file paths."""
    return [
        "data/dataset/0/1.jpg",
        "data/dataset/0/2.jpg",
        "data/dataset/1/3.jpg",
    ]


@pytest.fixture
def class_names() -> Dict[str, str]:
    """Class names mapping for testing."""
    return {
        "0": "bunny",
        "1": "impostor",
    }


@pytest.fixture
def dummy_embedded_images() -> np.ndarray:
    """Dummy embedded images.

    Returns:
        np.ndarray: of 3 images by 5 dimensions
    """
    return np.array([
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ])


@pytest.fixture
def expected_dataframe() -> pd.DataFrame:
    """Construct a DataFrame expected in embedding tests."""
    return pd.DataFrame([
        {
            "id": "1",
            "class_id": "0",
            "class_name": "bunny",
            "embedding": np.array([-0.1, -0.2, -0.3, -0.4, -0.5]),
        },
        {
            "id": "2",
            "class_id": "0",
            "class_name": "bunny",
            "embedding": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        },
        {
            "id": "3",
            "class_id": "1",
            "class_name": "impostor",
            "embedding": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        }
    ])


@pytest.fixture
def mock_terminator_embedding(
        dummy_file_paths: List[str],
        dummy_embedded_images: np.ndarray
) -> TerminatorEmbedding:
    """Construct a partially mocked embedding instance."""
    with (
        mock.patch("server.embedding.load_images") as mock_load_images,
        mock.patch("server.embedding.PCA") as mock_pca,
    ):
        mock_collection = mock_load_images.return_value
        mock_collection.__len__.return_value = 3
        mock_collection.files = dummy_file_paths
        mock_embedding = mock_pca.return_value
        mock_embedding.fit_transform.return_value = dummy_embedded_images
        embedding = TerminatorEmbedding("dummy")
        yield embedding


def test_get_vector_database(
        dummy_file_paths: List[str],
        dummy_embedded_images: np.ndarray,
        class_names: Dict[str, str],
        expected_dataframe: pd.DataFrame,
):
    """Test creating a pandas based 'vector database'."""
    got_df = get_vector_database(dummy_file_paths, dummy_embedded_images, class_names)
    pd.testing.assert_frame_equal(expected_dataframe, got_df)


def test_terminator_embedding_constructor(
        mock_terminator_embedding: TerminatorEmbedding,
        expected_dataframe: pd.DataFrame,
):
    """Test TerminatorEmbedding being constructed correctly."""
    # We expect the dataframe within the embedding instance to be equal to the manually constructed one
    pd.testing.assert_frame_equal(expected_dataframe, mock_terminator_embedding.dataframe)


def test_get_distance(
        mock_terminator_embedding: TerminatorEmbedding,
        expected_dataframe: pd.DataFrame
):
    """Test distance calculation."""
    expected_distances = [  # Approximately
        1.161895,
        0.38729833,
        0.0
    ]
    expected_distance_df = expected_dataframe
    expected_dataframe["distance"] = expected_distances
    distance_df = mock_terminator_embedding.get_distance("3")
    pd.testing.assert_frame_equal(expected_distance_df, distance_df)


def test_get_n_most_similar(
        mock_terminator_embedding: TerminatorEmbedding,
        expected_dataframe: pd.DataFrame
):
    """Test getting N most similar images."""
    similar_df = mock_terminator_embedding.get_n_most_similar("3", 3, "0")
    expected_distances = [  # Approximately
        0.38729833,
        1.161895,
    ]
    expected_similar_df = expected_dataframe.iloc[[1, 0]]
    expected_similar_df["distance"] = expected_distances
    pd.testing.assert_frame_equal(expected_similar_df, similar_df)