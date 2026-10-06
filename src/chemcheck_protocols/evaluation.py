"""Evaluation of signature quality: nearest-neighbour recapitulation (Comajuncosa-Creus et al. 2025, Fig. 3)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import paired_cosine_distances

logger = logging.getLogger(__name__)


SIGNATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("sign0", "sign1"),
    ("sign0", "sign2"),
    ("sign0", "sign3"),
    ("sign1", "sign2"),
    ("sign1", "sign3"),
    ("sign2", "sign3"),
)


def background_distance_cutoff(
    vectors: np.ndarray,
    p_value_cutoff: float = 0.01,
    n_pairs: int = 10000,
    n_subsamples: int = 10,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Cosine-distance cutoff defining "nearest neighbor" at a given p-value.

    This follows the CC Protocols paper's stated procedure: the cutoff is
    estimated **once, from a background distribution of randomly sampled
    compound pairs** (~10,000 pairs x 10 subsamples), and then applied as a
    fixed absolute distance threshold.

    The distinction matters. Taking the ``p_value_cutoff`` quantile *within*
    each evaluation subsample instead would force the positive rate to be
    exactly ``p_value_cutoff`` in every subsample and in both
    directions of a bidirectional comparison. That throws away the real
    signal of how many pairs are genuinely close in each space and makes the
    two directions artificially symmetric. A fixed background threshold lets
    the positive count vary as a property of the data, which is the point.

    Parameters
    ----------
    vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors of the space whose distance distribution defines the cutoff.
    p_value_cutoff : float, default 0.01
        Lower-tail probability of the background pairwise cosine-distance
        distribution used as the nearest-neighbor cutoff.
    n_pairs : int, default 10000
        Random compound pairs drawn per subsample.
    n_subsamples : int, default 10
        Number of subsamples the cutoff estimate is averaged over.
    random_state : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    tuple of (float, float)
        Mean and standard deviation of the cutoff across subsamples. The
        std is a stability diagnostic -- a large value means the background
        distribution was not sampled densely enough.

    Raises
    ------
    ValueError
        If fewer than two vectors are supplied.
    """
    n = vectors.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 vectors to sample pairs from, got {n}.")
    rng = np.random.default_rng(random_state)
    cutoffs = []
    for _ in range(n_subsamples):
        left = rng.integers(0, n, n_pairs)
        right = rng.integers(0, n, n_pairs)
        keep = left != right
        distances = paired_cosine_distances(vectors[left[keep]], vectors[right[keep]])
        cutoffs.append(np.quantile(distances, p_value_cutoff))
    return float(np.mean(cutoffs)), float(np.std(cutoffs))


def _recapitulation_subsamples(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    distance_cutoff: float,
    n_random: int,
    n_subsamples: int,
    random_state: int | None,
    fpr_grid: np.ndarray | None = None,
) -> tuple[list[float], list[np.ndarray], list[float]]:
    """Run the per-subsample recapitulation loop shared by the scalar/curve APIs.

    For each of ``n_subsamples`` draws of ``n_random`` compounds, every
    within-draw pair whose ``ref_vectors`` cosine distance is at or below
    ``distance_cutoff`` is labeled a positive, and the negated ``query_vectors``
    cosine distance for the same pair is used as the ranking score.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors. Nearest neighbors are defined on
        ``ref_vectors``; ``query_vectors`` is scored on recovering them.
    distance_cutoff : float
        Absolute cosine-distance threshold from
        :func:`background_distance_cutoff`.
    n_random : int
        Compounds sampled per repetition.
    n_subsamples : int
        Number of repetitions.
    random_state : int, optional
        Seed for reproducible subsampling.
    fpr_grid : numpy.ndarray, optional
        If given, each subsample's ROC curve is interpolated onto this grid
        and returned; otherwise no curves are collected.

    Returns
    -------
    tuple of (list of float, list of numpy.ndarray, list of float)
        Per-subsample AUROCs, interpolated TPR curves (empty when
        ``fpr_grid`` is None), and positive-pair rates.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate (single-class) label set.
    """
    rng = np.random.default_rng(random_state)
    n = ref_vectors.shape[0]
    n_random = min(n_random, n)
    aurocs: list[float] = []
    tprs: list[np.ndarray] = []
    pos_rates: list[float] = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=n_random, replace=False)
        d_ref = squareform(pdist(ref_vectors[idx], metric="cosine"))
        d_query = squareform(pdist(query_vectors[idx], metric="cosine"))
        iu = np.triu_indices_from(d_ref, k=1)
        y_true = (d_ref[iu] <= distance_cutoff).astype(int)
        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            logger.warning(
                "Degenerate subsample: %d/%d pairs fell within the cutoff "
                "(%.5f); skipping it.",
                n_pos,
                len(y_true),
                distance_cutoff,
            )
            continue
        y_score = -d_query[iu]
        pos_rates.append(n_pos / len(y_true))
        if fpr_grid is None:
            aurocs.append(float(roc_auc_score(y_true, y_score)))
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            tprs.append(np.interp(fpr_grid, fpr, tpr))
            aurocs.append(float(auc(fpr, tpr)))
    if not aurocs:
        raise RuntimeError(
            f"All {n_subsamples} subsamples were degenerate at cutoff {distance_cutoff:.5f}. "
            "Try a larger n_random or a less extreme p_value_cutoff."
        )
    return aurocs, tprs, pos_rates


def cosine_nn_recapitulation_auroc(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    p_value_cutoff: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> dict[str, float]:
    """AUROC for recapitulating reference-space nearest neighbors in a query space.

    Implements the recapitulation test of Comajuncosa-Creus et al. (Fig. 3a):
    compound pairs that are nearest neighbors in ``ref_vectors`` -- cosine
    distance at or below the ``p_value_cutoff`` cutoff of the *background* pairwise
    distance distribution -- are positives, and the ``query_vectors`` cosine
    distance for the same pairs is the ranking score. Repeated over
    ``n_subsamples`` draws of ``n_random`` compounds and averaged.

    Parameters
    ----------
    ref_vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors defining ground-truth nearest neighbors.
    query_vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors evaluated on their ability to recapitulate those neighbors.
        Must be row-aligned with ``ref_vectors``.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the NN cutoff.
    n_random : int, default 2500
        Compounds subsampled per repetition (the paper's value).
    n_subsamples : int, default 5
        Repetitions to average over (the paper's value).
    random_state : int, optional
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        ``auroc``, ``std``, ``cutoff``, ``cutoff_std``, ``mean_positive_rate``
        and ``n_valid_subsamples``. ``mean_positive_rate`` is reported because
        under a fixed background cutoff it is a real property of the space
        pair rather than a constant pinned to ``p_value_cutoff``.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate label set.
    """
    distance_cutoff, distance_cutoff_std = background_distance_cutoff(
        ref_vectors, p_value_cutoff=p_value_cutoff, random_state=random_state
    )
    logger.debug(
        "NN cutoff at pval=%.4g: cosine distance <= %.5f (+/- %.5f across "
        "background subsamples)",
        p_value_cutoff,
        distance_cutoff,
        distance_cutoff_std,
    )
    aurocs, _, pos_rates = _recapitulation_subsamples(
        ref_vectors,
        query_vectors,
        distance_cutoff,
        n_random,
        n_subsamples,
        random_state,
    )
    return {
        "auroc": float(np.mean(aurocs)),
        "std": float(np.std(aurocs)),
        "cutoff": distance_cutoff,
        "cutoff_std": distance_cutoff_std,
        "mean_positive_rate": float(np.mean(pos_rates)),
        "n_valid_subsamples": len(aurocs),
    }


def heldout_nn_recapitulation_auroc(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    heldout_mask: np.ndarray,
    p_value_cutoff: float = 0.01,
    n_bootstrap: int = 100,
    random_state: int | None = None,
) -> dict[str, Any]:
    """AUROC for recapitulating the reference neighbours of held-out molecules.

    The recapitulation test of :func:`cosine_nn_recapitulation_auroc`, restricted
    to pairs between a held-out molecule (left out when the space was fitted)
    and every other molecule. Pairs whose ``ref_vectors`` cosine distance is at
    or below the background ``p_value_cutoff`` cutoff are positives, and the
    ``query_vectors`` cosine distance ranks them. The uncertainty comes from
    bootstrapping the held-out molecules, since pairs sharing a molecule are not
    independent. With the same ``random_state``, two query spaces scored against
    the same reference get the same resamples, so their ``bootstrap_aurocs`` can
    be compared pairwise.

    Parameters
    ----------
    ref_vectors : numpy.ndarray, shape (n_molecules, n_features)
        Vectors defining the true neighbours, e.g. sign0 from a fit on all molecules.
    query_vectors : numpy.ndarray, shape (n_molecules, n_features)
        Vectors evaluated, e.g. sign3 of a space fitted without the held-out
        molecules. Must be row-aligned with ``ref_vectors``.
    heldout_mask : numpy.ndarray of bool, shape (n_molecules,)
        True for the held-out molecules.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the NN cutoff.
    n_bootstrap : int, default 100
        Bootstrap resamples of the held-out molecules.
    random_state : int, optional
        Seed for the background cutoff and the bootstrap.

    Returns
    -------
    dict
        ``auroc``, bootstrap ``std``, ``ci_low``/``ci_high`` (95%), ``cutoff``,
        ``cutoff_std``, ``n_heldout``, ``n_molecules``, ``n_pairs``,
        ``positive_rate`` and ``bootstrap_aurocs`` (NaN for degenerate resamples).

    Raises
    ------
    ValueError
        If the inputs are not row-aligned or no held-out molecule remains.
    RuntimeError
        If no pair (or every pair) falls within the cutoff.
    """
    heldout_mask = np.asarray(heldout_mask, dtype=bool)
    if not ref_vectors.shape[0] == query_vectors.shape[0] == heldout_mask.shape[0]:
        raise ValueError(
            f"Row counts differ: ref {ref_vectors.shape[0]}, query "
            f"{query_vectors.shape[0]}, heldout_mask {heldout_mask.shape[0]}"
        )
    # Cosine distance is undefined for all-zero vectors (e.g. inactive molecules in sign0).
    defined = (np.linalg.norm(ref_vectors, axis=1) > 0) & (
        np.linalg.norm(query_vectors, axis=1) > 0
    )
    if not defined.all():
        logger.warning(
            "Leaving out %d molecules with an all-zero vector (%d of them held out)",
            np.sum(~defined),
            np.sum(~defined & heldout_mask),
        )
    ref_vectors, query_vectors = ref_vectors[defined], query_vectors[defined]
    heldout_idx = np.flatnonzero(heldout_mask[defined])
    if len(heldout_idx) == 0:
        raise ValueError("No held-out molecules to evaluate")

    distance_cutoff, distance_cutoff_std = background_distance_cutoff(
        ref_vectors, p_value_cutoff=p_value_cutoff, random_state=random_state
    )
    ref_distances = cdist(ref_vectors[heldout_idx], ref_vectors, metric="cosine")
    query_distances = cdist(query_vectors[heldout_idx], query_vectors, metric="cosine")
    not_self = np.arange(ref_vectors.shape[0])[None, :] != heldout_idx[:, None]
    is_neighbour = ref_distances <= distance_cutoff

    def score(rows: np.ndarray) -> float:
        y_true = is_neighbour[rows][not_self[rows]]
        if y_true.all() or not y_true.any():
            return float("nan")
        return float(roc_auc_score(y_true, -query_distances[rows][not_self[rows]]))

    n_heldout = len(heldout_idx)
    auroc = score(np.arange(n_heldout))
    if np.isnan(auroc):
        raise RuntimeError(
            f"No informative pairs at cutoff {distance_cutoff:.5f}: "
            f"{is_neighbour[not_self].sum()} of {not_self.sum()} pairs are neighbours"
        )
    rng = np.random.default_rng(random_state)
    bootstrap_aurocs = np.array(
        [score(rng.integers(0, n_heldout, n_heldout)) for _ in range(n_bootstrap)]
    )
    ci_low, ci_high = np.nanpercentile(bootstrap_aurocs, [2.5, 97.5])
    return {
        "auroc": auroc,
        "std": float(np.nanstd(bootstrap_aurocs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "cutoff": distance_cutoff,
        "cutoff_std": distance_cutoff_std,
        "n_heldout": n_heldout,
        "n_molecules": int(ref_vectors.shape[0]),
        "n_pairs": int(not_self.sum()),
        "positive_rate": float(is_neighbour[not_self].mean()),
        "bootstrap_aurocs": bootstrap_aurocs,
    }


def pooled_heldout_auroc(
    ref_vectors: np.ndarray,
    heldout_rows: Sequence[int],
    query_vectors: Mapping[str, np.ndarray | Sequence[np.ndarray]],
    p_value_cutoff: float = 0.01,
    n_bootstrap: int = 100,
    random_state: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Held-out neighbour AUROC of several runs, with bootstrap resamples shared between them.

    The test of :func:`heldout_nn_recapitulation_auroc`, generalised in two ways: several
    runs are scored on the same pairs and the same resamples, so their differences are
    paired; and a run may be a set of fold models, each held-out molecule scored with the
    model that never saw it, all folds pooled into one AUROC. Each held-out molecule is
    paired with every other molecule; pairs whose ``ref_vectors`` cosine distance is at or
    below the background ``p_value_cutoff`` cutoff are positives, and each run's cosine
    distance ranks them. Resamples draw held-out molecules with replacement, since pairs
    sharing a molecule are not independent.

    Parameters
    ----------
    ref_vectors : numpy.ndarray, shape (n_molecules, n_features)
        Vectors defining the true neighbours, e.g. sign0 of a fit on all molecules.
    heldout_rows : sequence of int
        Rows of the held-out molecules, in the order the resamples index them.
    query_vectors : mapping of str to numpy.ndarray or sequence of numpy.ndarray
        Per run, either one matrix row-aligned with ``ref_vectors`` (a single model), or
        one matrix per held-out row (fold models: the matrix of the model that held out
        that molecule).
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the neighbour cutoff.
    n_bootstrap : int, default 100
        Resamples of the held-out molecules.
    random_state : int, optional
        Seed for the background cutoff and the resamples.

    Returns
    -------
    dict of str to dict
        Per run: ``auroc``, ``ci_low``/``ci_high`` (95%, percentile bootstrap), ``std``,
        ``bootstrap_aurocs`` (NaN for degenerate resamples), ``cutoff``, ``n_heldout``,
        ``n_molecules``, ``n_pairs`` and ``positive_pairs``.

    Raises
    ------
    ValueError
        If there are no held-out rows, a run's per-row matrices do not match
        ``heldout_rows``, or no held-out pair is a neighbour.
    """
    heldout_rows = np.asarray(heldout_rows, dtype=int)
    if len(heldout_rows) == 0:
        raise ValueError("No held-out molecules to evaluate")
    n_molecules = ref_vectors.shape[0]
    cutoff, _ = background_distance_cutoff(
        ref_vectors, p_value_cutoff=p_value_cutoff, random_state=random_state
    )
    others = [np.arange(n_molecules) != row for row in heldout_rows]
    labels = [
        cdist(ref_vectors[[row]], ref_vectors, metric="cosine")[0, keep] <= cutoff
        for row, keep in zip(heldout_rows, others)
    ]
    if not any(label.any() for label in labels):
        raise ValueError(f"No held-out pair is a neighbour at cutoff {cutoff:.5f}")
    distances = {}
    for run, vectors in query_vectors.items():
        per_row = (
            [vectors] * len(heldout_rows)
            if isinstance(vectors, np.ndarray)
            else list(vectors)
        )
        if len(per_row) != len(heldout_rows):
            raise ValueError(
                f"Run {run!r} has {len(per_row)} matrices for {len(heldout_rows)} held-out rows"
            )
        distances[run] = [
            cdist(matrix[[row]], matrix, metric="cosine")[0, keep]
            for matrix, row, keep in zip(per_row, heldout_rows, others)
        ]

    def score(run: str, molecules: np.ndarray) -> float:
        y_true = np.concatenate([labels[m] for m in molecules])
        if y_true.all() or not y_true.any():
            return float("nan")
        return float(
            roc_auc_score(
                y_true, -np.concatenate([distances[run][m] for m in molecules])
            )
        )

    n_heldout = len(heldout_rows)
    rng = np.random.default_rng(random_state)
    resamples = [rng.integers(0, n_heldout, n_heldout) for _ in range(n_bootstrap)]
    results = {}
    for run in query_vectors:
        bootstrap = np.array([score(run, molecules) for molecules in resamples])
        ci_low, ci_high = np.nanpercentile(bootstrap, [2.5, 97.5])
        results[run] = {
            "auroc": score(run, np.arange(n_heldout)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "std": float(np.nanstd(bootstrap)),
            "bootstrap_aurocs": bootstrap,
            "cutoff": cutoff,
            "n_heldout": n_heldout,
            "n_molecules": int(n_molecules),
            "n_pairs": int(sum(len(label) for label in labels)),
            "positive_pairs": int(sum(label.sum() for label in labels)),
        }
    logger.info(
        "Scored %d runs on %d held-out molecules (%d neighbour pairs)",
        len(results),
        n_heldout,
        results[next(iter(results))]["positive_pairs"],
    )
    return results


def paired_difference(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float]:
    """
    Difference ``first - second`` of two runs scored on the same resamples, with its 95% CI.

    Parameters
    ----------
    first, second : mapping
        Results of :func:`pooled_heldout_auroc` (or :func:`heldout_nn_recapitulation_auroc`
        with the same seed and molecules): ``auroc`` and ``bootstrap_aurocs``.

    Returns
    -------
    dict of str to float
        ``difference``, ``ci_low``/``ci_high`` (percentiles of the per-resample differences)
        and ``fraction_first_better`` (share of resamples where ``first`` scores higher).
    """
    deltas = np.asarray(first["bootstrap_aurocs"]) - np.asarray(
        second["bootstrap_aurocs"]
    )
    ci_low, ci_high = np.nanpercentile(deltas, [2.5, 97.5])
    return {
        "difference": float(first["auroc"] - second["auroc"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fraction_first_better": float(np.nanmean(deltas > 0)),
    }


def per_molecule_heldout_auroc(
    ref_vectors: np.ndarray,
    heldout_rows: Sequence[int],
    query_vectors: Mapping[str, np.ndarray],
    p_value_cutoff: float = 0.01,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Held-out neighbour AUROC of each held-out molecule on its own pairs, per run.

    Shows whether a difference between runs is broad or carried by a few molecules. A
    molecule can only be scored if it has at least one neighbour (and one non-neighbour)
    among its pairs; otherwise its AUROC is NaN.

    Parameters
    ----------
    ref_vectors : numpy.ndarray, shape (n_molecules, n_features)
        Vectors defining the true neighbours.
    heldout_rows : sequence of int
        Rows of the held-out molecules.
    query_vectors : mapping of str to numpy.ndarray
        Per run, a matrix row-aligned with ``ref_vectors``.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the neighbour cutoff.
    random_state : int, optional
        Seed for the background cutoff.

    Returns
    -------
    pandas.DataFrame
        One row per held-out molecule: ``row``, ``n_neighbours`` and one AUROC column per run.
    """
    cutoff, _ = background_distance_cutoff(
        ref_vectors, p_value_cutoff=p_value_cutoff, random_state=random_state
    )
    rows = []
    for row in heldout_rows:
        others = np.arange(ref_vectors.shape[0]) != row
        is_neighbour = (
            cdist(ref_vectors[[row]], ref_vectors, metric="cosine")[0, others] <= cutoff
        )
        record = {"row": int(row), "n_neighbours": int(is_neighbour.sum())}
        informative = 0 < is_neighbour.sum() < len(is_neighbour)
        for run, vectors in query_vectors.items():
            distances = cdist(vectors[[row]], vectors, metric="cosine")[0, others]
            record[run] = (
                roc_auc_score(is_neighbour, -distances) if informative else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def get_shared_vectors(
    signature_a,
    signature_b,
    max_pool: int = 50000,
    random_state: int | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fetch row-aligned sign3 vectors for compounds shared by two spaces.

    Two sign3 spaces each cover essentially the whole ~1.2M-compound CC
    universe, so the shared key set is near-total. Materializing it in full
    for both spaces costs roughly ``2 x 1.2e6 x 128 x 4 bytes`` = **1.2 GB**
    of resident memory, enough to make plotting jobs that run alongside it
    fail (observed as truncated, zero-byte figures).

    Nothing downstream needs the full set. The recapitulation test samples
    ``n_random`` (2,500) compounds per repetition and the background cutoff
    samples 10,000 pairs, so a random pool of ``max_pool`` shared compounds
    is statistically equivalent and roughly 25x cheaper. The pool is drawn
    once and shared by both spaces so the two vector arrays stay aligned.

    Parameters
    ----------
    signature_a, signature_b : chemicalchecker.core.signature_data.DataSignature
        Fitted ``sign3`` signature objects for the two spaces being compared.
    max_pool : int, default 50000
        Cap on the number of shared compounds fetched. Pass ``0`` to disable
        subsampling and fetch every shared key (memory-hungry; see above).
    random_state : int, optional
        Seed for reproducible pool subsampling.

    Returns
    -------
    tuple of (list of str, numpy.ndarray, numpy.ndarray)
        The pooled shared keys, and their vectors from space A and space B,
        row-aligned to each other.

    Raises
    ------
    ValueError
        If fewer than 50 compounds are shared, or ``get_vectors()`` returns
        no rows.
    RuntimeError
        If the two spaces' ``get_vectors()`` calls return keys in different
        order (should not happen for an identical input key set).
    """
    shared = sorted(set(signature_a.keys) & set(signature_b.keys))
    n_shared = len(shared)
    if n_shared < 50:
        raise ValueError(
            f"Only {n_shared} compounds shared between the two spaces; "
            "too few for a reliable recapitulation estimate."
        )

    pool = shared
    if max_pool and n_shared > max_pool:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_shared, size=max_pool, replace=False)
        # Keep sorted order: get_vectors() returns keys sorted, and sorting
        # here keeps the returned pool list in the same order as the vectors.
        pool = [shared[i] for i in np.sort(idx)]
        logger.info(
            "%d shared compounds; sampling a pool of %d for the recapitulation "
            "test (max_pool=0 uses all, at ~1.2 GB peak memory).",
            n_shared,
            max_pool,
        )
    else:
        logger.info(
            "Using all %d shared compounds for the recapitulation test.", n_shared
        )

    # DataSignature.__getitem__ only supports integer fancy indexing
    # (internally slice(min(key), max(key)+1)), which silently breaks for a
    # list of InChIKey strings -- get_vectors() is the library's own
    # fetch-rows-by-key method and handles this correctly. It returns
    # (sorted_keys_found, vectors); both calls query the same key set, so
    # the returned orders match and the arrays are already row-aligned.
    inks_a, vectors_a = signature_a.get_vectors(pool)
    inks_b, vectors_b = signature_b.get_vectors(pool)
    if vectors_a is None or vectors_b is None:
        raise ValueError("get_vectors() returned no rows for the shared compound set.")
    if not np.array_equal(inks_a, inks_b):
        raise RuntimeError(
            "Key order mismatch between the two spaces' get_vectors() results -- "
            "this should not happen for an identical input key set; inspect "
            "get_vectors() behavior in your chemicalchecker version."
        )
    return list(inks_a), vectors_a, vectors_b


def shared_key_recapitulation(
    vectors_a: np.ndarray,
    vectors_b: np.ndarray,
    n_shared_compounds: int,
    p_value_cutoff: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Bidirectional NN-recapitulation AUROC over the shared compound set.

    Runs the test in both directions -- how well B recovers A's neighbor
    structure, and vice versa -- because the two are not equivalent once the
    NN cutoff is a fixed background threshold rather than a per-subsample
    quantile. Asymmetry between the directions is itself informative: it
    says one space's neighbor structure is the more reproducible of the two.

    This operates over the near-total CC-universe key intersection, matching
    the paper's Ext. Data Fig. 8g,h methodology for comparing two *different*
    datasets' type III signatures. It deliberately does **not** restrict to
    either dataset's real input compounds -- that is a different question,
    answered by :func:`plot_input_compound_overlay`.

    Parameters
    ----------
    vectors_a, vectors_b : numpy.ndarray
        Row-aligned sign3 vectors (see :func:`get_shared_vectors`).
    n_shared_compounds : int
        Total number of shared compounds, recorded for the report. This is
        the full intersection size, which may exceed ``len(vectors_a)`` when a
        pool subsample was taken.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the NN cutoff.
    n_random : int, default 2500
        Compounds subsampled per repetition.
    n_subsamples : int, default 5
        Repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        ``n_shared_compounds``, ``n_pool_compounds``, ``pval``, and an
        ``a_recap_by_b`` / ``b_recap_by_a`` sub-dict each holding the full
        output of :func:`cosine_nn_recapitulation_auroc`.
    """
    a_by_b = cosine_nn_recapitulation_auroc(
        vectors_a,
        vectors_b,
        p_value_cutoff=p_value_cutoff,
        n_random=n_random,
        n_subsamples=n_subsamples,
        random_state=random_state,
    )
    b_by_a = cosine_nn_recapitulation_auroc(
        vectors_b,
        vectors_a,
        p_value_cutoff=p_value_cutoff,
        n_random=n_random,
        n_subsamples=n_subsamples,
        random_state=random_state,
    )
    return {
        "n_shared_compounds": int(n_shared_compounds),
        "n_pool_compounds": int(vectors_a.shape[0]),
        "pval": p_value_cutoff,
        "a_recap_by_b": a_by_b,
        "b_recap_by_a": b_by_a,
    }


def recapitulation_roc_band(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    p_value_cutoff: float,
    n_random: int,
    n_subsamples: int,
    random_state: int | None,
    fpr_grid: np.ndarray,
) -> dict[str, Any]:
    """Mean +/- std ROC curve across subsamples, on a common FPR grid.

    Same methodology as :func:`cosine_nn_recapitulation_auroc`, but keeps
    each subsample's full ROC curve (interpolated onto ``fpr_grid``) instead
    of collapsing straight to a scalar. This is what reproduces the
    shaded-band ROC plots of Fig. 3g,h rather than just the number in
    Fig. 3b.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors; NN pairs are defined on ``ref_vectors`` and
        recapitulation is scored against ``query_vectors``.
    p_value_cutoff : float
        Background-distribution p-value defining the NN cutoff.
    n_random : int
        Compounds subsampled per repetition.
    n_subsamples : int
        Repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.
    fpr_grid : numpy.ndarray
        Common FPR grid each subsample's ROC curve is interpolated onto.

    Returns
    -------
    dict
        ``mean_tpr``, ``std_tpr`` (arrays on ``fpr_grid``), ``auroc``,
        ``std``, ``cutoff`` and ``mean_positive_rate``.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate label set.
    """
    distance_cutoff, _ = background_distance_cutoff(
        ref_vectors, p_value_cutoff=p_value_cutoff, random_state=random_state
    )
    aurocs, tprs, pos_rates = _recapitulation_subsamples(
        ref_vectors,
        query_vectors,
        distance_cutoff,
        n_random,
        n_subsamples,
        random_state,
        fpr_grid,
    )
    tprs_arr = np.vstack(tprs)
    return {
        "mean_tpr": tprs_arr.mean(axis=0),
        "std_tpr": tprs_arr.std(axis=0),
        "auroc": float(np.mean(aurocs)),
        "std": float(np.std(aurocs)),
        "cutoff": distance_cutoff,
        "mean_positive_rate": float(np.mean(pos_rates)),
    }


def get_signature_type_vectors(
    cc, dataset_code: str, sign_types: tuple[str, ...]
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Fetch row-aligned vectors for several signature types of one dataset.

    Restricts to the compounds present in *every* requested signature type.
    In practice that is the dataset's sign0 key set: sign0/1/2 hold exactly
    the real input compounds while sign3 spans the whole CC universe, so the
    intersection is the set of molecules the dataset actually measured --
    which is the right population for a within-space comparison.

    Rows whose vector has zero norm in any signature type are dropped:
    cosine distance is undefined for them and would propagate NaN into every
    downstream AUROC.

    Parameters
    ----------
    cc : chemicalchecker.core.chemcheck.ChemicalChecker
        CC instance the dataset belongs to.
    dataset_code : str
        CC dataset code, e.g. ``"M1.001"``.
    sign_types : tuple of str
        Signature types to fetch, e.g. ``("sign0", "sign1", "sign2", "sign3")``.

    Returns
    -------
    tuple of (list of str, dict)
        The common keys, and a mapping of signature type to its row-aligned
        vector array.

    Raises
    ------
    ValueError
        If fewer than 50 compounds are common to all requested types.
    """
    signatures = {st: cc.get_signature(st, "full", dataset_code) for st in sign_types}
    common: set[str] | None = None
    for st, sig in signatures.items():
        ks = set(map(str, sig.keys))
        common = ks if common is None else (common & ks)
    keys = sorted(common or set())

    vectors: dict[str, np.ndarray] = {}
    for st, sig in signatures.items():
        found, vec = sig.get_vectors(keys)
        if vec is None:
            raise ValueError(f"{dataset_code} {st}: get_vectors() returned no rows.")
        keys = [str(k) for k in found]
        vectors[st] = np.asarray(vec, dtype=np.float64)

    # Guard against zero-norm rows before any cosine distance is taken.
    good = np.ones(len(keys), dtype=bool)
    for st, vec in vectors.items():
        norms = np.linalg.norm(vec, axis=1)
        zero = norms == 0
        if zero.any():
            logger.warning(
                "%s %s: dropping %d compound(s) with a zero-norm vector "
                "(cosine distance undefined).",
                dataset_code,
                st,
                int(zero.sum()),
            )
        good &= ~zero
    if not good.all():
        keys = [k for k, ok in zip(keys, good) if ok]
        vectors = {st: vec[good] for st, vec in vectors.items()}

    if len(keys) < 50:
        raise ValueError(
            f"{dataset_code}: only {len(keys)} compounds common to "
            f"{list(sign_types)}; too few for a recapitulation estimate."
        )
    logger.info(
        "%s: %d compounds common to %s.", dataset_code, len(keys), list(sign_types)
    )
    return keys, vectors


def signature_recovery(
    cc,
    dataset_code: str,
    label: str,
    pairs: tuple[tuple[str, str], ...] = SIGNATURE_PAIRS,
    p_value_cutoff: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Within-space signature recovery across signature types (Fig. 3b).

    For each ``(lower, higher)`` pair, nearest-neighbour compound pairs are
    defined at the *lower* signature type using the background-distribution
    cutoff at ``p_value_cutoff``, and the *higher* type is scored by AUROC on
    recovering them. This is the paper's own check that abstraction from
    sign0 up to sign3 preserves the similarity structure of the raw data;
    running it for two spaces side by side shows whether one of them loses
    more of its own structure on the way up.

    Note on subsampling: the paper draws 2,500 molecules per repetition from
    a larger pool. These datasets have fewer input compounds than that, so
    drawing ``min(n_random, n)`` would make every repetition the identical
    full set and report a meaningless ``std`` of exactly zero. When the pool
    is the limiting factor, 80% of it is drawn per repetition instead, so
    the reported spread reflects real sampling variability.

    Parameters
    ----------
    cc : chemicalchecker.core.chemcheck.ChemicalChecker
        CC instance the dataset belongs to.
    dataset_code : str
        CC dataset code, e.g. ``"M1.001"``.
    label : str
        Display label for the space, carried into the returned frame.
    pairs : tuple of (str, str), default SIGNATURE_PAIRS
        ``(lower, higher)`` signature-type combinations to evaluate.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the NN cutoff.
    n_random : int, default 2500
        Compounds drawn per repetition, capped by the available pool.
    n_subsamples : int, default 5
        Repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.

    Returns
    -------
    pandas.DataFrame
        One row per pair: ``space``, ``dataset_code``, ``pair``, ``lower``,
        ``higher``, ``auroc``, ``std``, ``cutoff``, ``mean_positive_rate``
        and ``n_compounds``.
    """
    sign_types = tuple(sorted({st for pair in pairs for st in pair}))
    keys, vectors = get_signature_type_vectors(cc, dataset_code, sign_types)
    n = len(keys)

    draw = min(n_random, n)
    if draw >= n and n_subsamples > 1:
        draw = max(50, int(0.8 * n))
        logger.info(
            "%s: pool of %d compounds is smaller than n_random=%d; drawing %d "
            "(80%%) per repetition so the reported std is meaningful.",
            dataset_code,
            n,
            n_random,
            draw,
        )

    rows = []
    for lower, higher in pairs:
        try:
            res = cosine_nn_recapitulation_auroc(
                vectors[lower],
                vectors[higher],
                p_value_cutoff=p_value_cutoff,
                n_random=draw,
                n_subsamples=n_subsamples,
                random_state=random_state,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "%s: skipping %s->%s recovery: %s", dataset_code, lower, higher, exc
            )
            continue
        rows.append(
            {
                "space": label,
                "dataset_code": dataset_code,
                "pair": f"{lower}\u2192{higher}",
                "lower": lower,
                "higher": higher,
                "auroc": res["auroc"],
                "std": res["std"],
                "cutoff": res["cutoff"],
                "mean_positive_rate": res["mean_positive_rate"],
                "n_compounds": n,
            }
        )
        logger.info(
            "%s %s->%s recovery: AUROC=%.3f +/- %.3f (n=%d)",
            dataset_code,
            lower,
            higher,
            res["auroc"],
            res["std"],
            n,
        )
    return pd.DataFrame(rows)
