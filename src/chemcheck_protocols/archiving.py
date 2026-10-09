"""Slim fitted Chemical Checker (CC) instances by removing files that ``fit`` can regenerate.

Each ``sign3`` keeps, in its ``models`` folder, the stacked sign2 of every reference space
(``all_sign2.h5``, ``all_sign2_coverage.h5`` and their ``.complete.h5`` copies, about 30 GiB).
They are read only while fitting (``fit``, ``train_SNN``, ``plot_validations``, ...); ``fit``
rebuilds them from the reference sign2 signatures, and reading ``sign3.h5`` or predicting with
the trained network does not need them. The training files (``train.h5``, ``traintest_*.h5``)
are kept: they record the triplets the network was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# <root>/<molecule set>/<coordinate letter>/<coordinate>/<dataset code>/sign3/models/: a CC
# instance keeps its signatures for the "full" molecule set and for the "reference" one.
MOLECULE_SETS = ("full", "reference")
REGENERABLE_SIGN3_PATTERN = "{molecule_set}/*/*/*/sign3/models/all_sign2*.h5"


def find_regenerable_files(cc_root: Path) -> list[Path]:
    """
    List the regenerable files of the fitted sign3 in a CC instance, in both molecule sets
    (``full/`` and ``reference/``).

    Symbolic links are skipped, as is anything that resolves outside ``cc_root``: spaces linked
    from a shared release (e.g. a CC update) must never be touched.

    Parameters
    ----------
    cc_root : pathlib.Path
        Root of the CC instance (the folder that contains ``full/``).

    Returns
    -------
    list of pathlib.Path
        The files, sorted.

    Raises
    ------
    ValueError
        If ``cc_root`` is not a folder with a ``full/`` subfolder.
    """
    if not (cc_root / "full").is_dir():
        raise ValueError(f"{cc_root} is not a CC instance: no full/ folder inside")
    real_root = cc_root.resolve()
    files = []
    candidates = sorted(
        path
        for molecule_set in MOLECULE_SETS
        for path in cc_root.glob(
            REGENERABLE_SIGN3_PATTERN.format(molecule_set=molecule_set)
        )
    )
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(real_root):
            logger.warning(
                "Skipping %s: it is a link or leads outside the instance", path
            )
            continue
        files.append(path)
    return files


def delete_files(files: list[Path]) -> int:
    """Delete ``files`` and return the number of bytes freed."""
    freed_bytes = 0
    for path in files:
        size = path.stat().st_size
        path.unlink()
        freed_bytes += size
        logger.info("Deleted %s (%s)", path, format_size(size))
    return freed_bytes


def format_size(size_bytes: int) -> str:
    """Format a byte count with a binary unit, e.g. ``1.5 GiB``."""
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
