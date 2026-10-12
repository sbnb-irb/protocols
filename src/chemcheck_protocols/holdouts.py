"""Held-out molecule sets for evaluation: a random fraction, disjoint folds, or folds stratified by group.

Every draw sorts the keys first and uses ``numpy.random.default_rng(seed)``, so the same keys and
seed always give the same sets, whatever order the keys came in.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def random_holdout(keys: Iterable[str], fraction: float, seed: int) -> list[str]:
    """
    Draw a random fraction of the keys.

    Parameters
    ----------
    keys : iterable of str
        The pool, e.g. the molecules that reach sign2 in a fit on all molecules.
    fraction : float
        Share of the pool to hold out, between 0 and 1 (exclusive).
    seed : int
        Seed of the random draw.

    Returns
    -------
    list of str
        ``round(fraction * len(pool))`` keys, sorted.

    Raises
    ------
    ValueError
        If ``fraction`` is not strictly between 0 and 1, or the draw would be empty.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    pool = np.array(sorted(set(keys)))
    size = round(fraction * len(pool))
    if size == 0:
        raise ValueError(
            f"{fraction} of {len(pool)} keys rounds to an empty held-out set"
        )
    rng = np.random.default_rng(seed)
    heldout = sorted(rng.choice(pool, size=size, replace=False).tolist())
    logger.info(
        "Held out %d of %d keys (fraction %g, seed %d)", size, len(pool), fraction, seed
    )
    return heldout


def disjoint_folds(keys: Iterable[str], n_folds: int, seed: int) -> list[list[str]]:
    """
    Split the keys into disjoint folds of (nearly) equal size.

    The sorted keys are shuffled once and cut into ``n_folds`` consecutive chunks, so fold
    ``i`` is the same whatever ``n_folds`` folds are used later (e.g. the first two folds of
    a five-fold split).

    Parameters
    ----------
    keys : iterable of str
        The pool to split.
    n_folds : int
        Number of folds, at least 2 and at most the number of keys.
    seed : int
        Seed of the shuffle.

    Returns
    -------
    list of list of str
        One sorted key list per fold.

    Raises
    ------
    ValueError
        If ``n_folds`` is below 2 or larger than the pool.
    """
    pool = np.array(sorted(set(keys)))
    if not 2 <= n_folds <= len(pool):
        raise ValueError(f"n_folds must be between 2 and {len(pool)}, got {n_folds}")
    shuffled = pool[np.random.default_rng(seed).permutation(len(pool))]
    folds = [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, n_folds)]
    logger.info(
        "Split %d keys into %d folds of %s (seed %d)",
        len(pool),
        n_folds,
        [len(f) for f in folds],
        seed,
    )
    return folds


def stratified_folds(
    groups: Mapping[str, str], n_folds: int, seed: int
) -> list[list[str]]:
    """
    Split keys into folds that each get a share of every group (e.g. study).

    Groups are taken in sorted order; within each group the sorted keys are shuffled and dealt
    to the folds in turn, continuing from where the previous group stopped, so fold sizes
    differ by at most one.

    Parameters
    ----------
    groups : mapping of str to str
        Group of each key.
    n_folds : int
        Number of folds, at least 2 and at most the number of keys.
    seed : int
        Seed of the shuffles.

    Returns
    -------
    list of list of str
        One sorted key list per fold.

    Raises
    ------
    ValueError
        If ``n_folds`` is below 2 or larger than the number of keys.
    """
    if not 2 <= n_folds <= len(groups):
        raise ValueError(f"n_folds must be between 2 and {len(groups)}, got {n_folds}")
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    position = 0
    for group in sorted(set(groups.values())):
        members = sorted(key for key, value in groups.items() if value == group)
        for key in rng.permutation(members):
            folds[position % n_folds].append(str(key))
            position += 1
    logger.info(
        "Split %d keys from %d groups into %d folds of %s (seed %d)",
        len(groups),
        len(set(groups.values())),
        n_folds,
        [len(f) for f in folds],
        seed,
    )
    return [sorted(fold) for fold in folds]


def write_key_list(
    path: str | os.PathLike[str], keys: Collection[str], comment: str = ""
) -> Path:
    """
    Write keys one per line, after an optional ``#`` comment, readable by ``load_key_list``.

    Parameters
    ----------
    path : str or os.PathLike
        The output file; its folder is created if needed.
    keys : collection of str
        The keys, written in the given order.
    comment : str, optional
        Description written as ``#`` lines at the top (one per line of ``comment``).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(f"# {line}\n" for line in comment.splitlines() if line.strip())
    path.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")
    logger.info("Wrote %d keys to %s", len(keys), path)
    return path
