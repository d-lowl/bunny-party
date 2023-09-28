import datetime
from unittest import mock
from contextlib import nullcontext as does_not_raise

import pytest

from server.dvc import DVCFileVersion


@pytest.fixture
def mock_commit() -> mock.MagicMock:
    """Mock a basic commit."""
    with mock.patch("git.Commit") as mock_commit_constructor:
        mock_commit = mock_commit_constructor.return_value
        mock_commit.committed_datetime = datetime.datetime(2010, 11, 15, 10, 10, 10)
        mock_commit.name_rev = "deadbeef"
        yield mock_commit


@pytest.fixture
def mock_commit_with_tag(mock_commit: mock.MagicMock) -> mock.MagicMock:
    """Mock a commit with a tag."""
    mock_commit.name_rev += " tags/v1"
    yield mock_commit


@pytest.fixture
def mock_git_repo(mock_commit: mock.MagicMock) -> mock.MagicMock:
    """Mock a git repo."""
    expected_git_hash = "deadbeef"
    with mock.patch("server.dvc.Repo") as mock_repo_constructor:
        mock_repo = mock_repo_constructor.return_value
        mock_repo.git_dir = "."
        mock_repo.git.rev_list.return_value = expected_git_hash
        mock_repo.commit.return_value = mock_commit
        yield mock_repo


@pytest.fixture
def mock_git_repo_with_tag(mock_git_repo: mock.MagicMock, mock_commit_with_tag: mock.MagicMock) -> mock.MagicMock:
    """Mock a git repo with a tagged commit."""
    mock_git_repo.commit.return_value = mock_commit_with_tag
    yield mock_git_repo


def test_from_filepath_not_tracked():
    """Should throw exception if the file is not tracked."""
    with mock.patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(Exception):
            version = DVCFileVersion.from_filepath("test/file/path")


def test_from_filepath_commit_hash_only(mock_git_repo: mock.MagicMock):
    """Test the DVC file version is constructed correctly."""
    expected = DVCFileVersion(
        filepath="test/file/path",
        dot_dvc_filepath="test/file/path.dvc",
        git_revisions=["deadbeef"],
        committed_datetime=datetime.datetime(2010, 11, 15, 10, 10, 10)
    )
    with mock.patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        with does_not_raise():
            version = DVCFileVersion.from_filepath("test/file/path")
            assert expected == version


def test_from_filepath_commit_with_tag(mock_git_repo_with_tag: mock.MagicMock):
    """Test the DVC file version is constructed correctly."""
    expected = DVCFileVersion(
        filepath="test/file/path",
        dot_dvc_filepath="test/file/path.dvc",
        git_revisions=["deadbeef", "tags/v1"],
        committed_datetime=datetime.datetime(2010, 11, 15, 10, 10, 10)
    )
    with mock.patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        with does_not_raise():
            version = DVCFileVersion.from_filepath("test/file/path")
            assert expected == version
