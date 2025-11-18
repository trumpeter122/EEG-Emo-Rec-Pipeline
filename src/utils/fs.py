"""Filesystem utilities to support atomic pipeline operations."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ["atomic_directory", "directory_is_populated"]


def directory_is_populated(path: Path) -> bool:
    """
    Return ``True`` if ``path`` exists and contains at least one entry.

    The helper treats non-existent directories as empty so callers can probe
    destinations before they are materialized on disk.
    """
    return path.exists() and any(path.iterdir())


@contextmanager
def atomic_directory(target_dir: Path) -> Iterator[Path]:
    """
    Yield a staging directory that atomically replaces ``target_dir`` on success.

    All writes must be performed inside the yielded directory.  When the context
    exits without errors the staging directory replaces ``target_dir``; if an
    exception occurs the staging directory is removed to avoid leaving partial
    artifacts on disk.
    """
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = parent_dir / f".{target_dir.name}.staging-{uuid4().hex}"
    staging_dir.mkdir()

    try:
        yield staging_dir
    except BaseException:
        shutil.rmtree(path=staging_dir, ignore_errors=True)
        raise

    if target_dir.exists():
        shutil.rmtree(path=target_dir)

    staging_dir.rename(target_dir)
