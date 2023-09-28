"""Module to get DVC specific information.

To date, DVC Python API lacks some things that I'd prefer it to have, such as programmatically getting file versions
(as commit hashes or git tags). Hence we query git directly with https://github.com/gitpython-developers/GitPython.
"""
import datetime
import os
from typing import List

from git import Repo
from pydantic import BaseModel
from typing_extensions import Self


class DVCFileVersion(BaseModel):
    filepath: str
    dot_dvc_filepath: str
    git_revisions: List[str]
    committed_datetime: datetime.datetime

    @classmethod
    def from_filepath(cls, filepath: str) -> Self:
        """Get verbose DVC version for a filepath

        Args:
            filepath (str):

        Returns:
            DVCFileVersion:

        Raises:
             Exception: if the file is not tracked by DVC
        """
        dot_dvc_filepath = filepath + ".dvc"
        if not os.path.exists(dot_dvc_filepath):
            raise Exception(f"{filepath} is not tracked by DVC")

        repo = Repo(".")
        commit_hash = repo.git.rev_list("-1", "HEAD", dot_dvc_filepath)
        commit = repo.commit(commit_hash)
        committed_datetime = commit.committed_datetime
        git_revisions = commit.name_rev.split(" ")

        # Cleanup, see GitPython limitations
        del commit
        del repo

        return cls(
            filepath=filepath,
            dot_dvc_filepath=dot_dvc_filepath,
            git_revisions=git_revisions,
            committed_datetime=committed_datetime
        )


