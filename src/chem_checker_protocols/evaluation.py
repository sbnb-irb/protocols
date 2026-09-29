"""Evaluation of signature quality: nearest-neighbour recapitulation (Comajuncosa-Creus et al. 2025, Fig. 3)."""

from __future__ import annotations

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import paired_cosine_distances
from typing import Any
import logging
import numpy as np
import pandas as pd

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
    pval: float = 0.01,
    n_pairs: int = 10000,
    n_subsamples: int = 10,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Cosine-distance cutoff defining "nearest neighbor" at a given p-value.

    This follows the CC Protocols paper's stated procedure: the cutoff is
    estimated **once, from a background distribution of randomly sampled
    compound pairs** (~10,000 pairs x 10 subsamples), and then applied as a
    fixed absolute distance threshold.

    The distinction matters. Taking the ``pval`` quantile *within* each
    evaluation subsample -- which is what this module used to do -- forces
    the positive rate to be exactly ``pval`` in every subsample and in both
    directions of a bidirectional comparison. That throws away the real
    signal of how many pairs are genuinely close in each space and makes the
    two directions artificially symmetric. A fixed background threshold lets
    the positive count vary as a property of the data, which is the point.

    Parameters
    ----------
    vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors of the space whose distance distribution defines the cutoff.
    pval : float, default 0.01
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
        cutoffs.append(np.quantile(distances, pval))
    return float(np.mean(cutoffs)), float(np.std(cutoffs))


def _recapitulation_subsamples(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    cutoff: float,
    n_random: int,
    n_subsamples: int,
    random_state: int | None,
    fpr_grid: np.ndarray | None = None,
) -> tuple[list[float], list[np.ndarray], list[float]]:
    """Run the per-subsample recapitulation loop shared by the scalar/curve APIs.

    For each of ``n_subsamples`` draws of ``n_random`` compounds, every
    within-draw pair whose ``ref_vectors`` cosine distance is at or below
    ``cutoff`` is labeled a positive, and the negated ``query_vectors``
    cosine distance for the same pair is used as the ranking score.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors. Nearest neighbors are defined on
        ``ref_vectors``; ``query_vectors`` is scored on recovering them.
    cutoff : float
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
        y_true = (d_ref[iu] <= cutoff).astype(int)
        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            logger.warning(
                "Degenerate subsample: %d/%d pairs fell within the cutoff "
                "(%.5f); skipping it.",
                n_pos,
                len(y_true),
                cutoff,
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
            f"All {n_subsamples} subsamples were degenerate at cutoff {cutoff:.5f}. "
            "Try a larger --n-random or a less extreme --pval."
        )
    return aurocs, tprs, pos_rates


def cosine_nn_recapitulation_auroc(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    pval: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> dict[str, float]:
    """AUROC for recapitulating reference-space nearest neighbors in a query space.

    Implements the recapitulation test of Comajuncosa-Creus et al. (Fig. 3a):
    compound pairs that are nearest neighbors in ``ref_vectors`` -- cosine
    distance at or below the ``pval`` cutoff of the *background* pairwise
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
    pval : float, default 0.01
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
        pair rather than a constant pinned to ``pval``.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate label set.
    """
    cutoff, cutoff_std = background_distance_cutoff(
        ref_vectors, pval=pval, random_state=random_state
    )
    logger.debug(
        "NN cutoff at pval=%.4g: cosine distance <= %.5f (+/- %.5f across "
        "background subsamples)",
        pval,
        cutoff,
        cutoff_std,
    )
    aurocs, _, pos_rates = _recapitulation_subsamples(
        ref_vectors, query_vectors, cutoff, n_random, n_subsamples, random_state
    )
    return {
        "auroc": float(np.mean(aurocs)),
        "std": float(np.std(aurocs)),
        "cutoff": cutoff,
        "cutoff_std": cutoff_std,
        "mean_positive_rate": float(np.mean(pos_rates)),
        "n_valid_subsamples": len(aurocs),
    }


def get_shared_vectors(
    sign3_a,
    sign3_b,
    max_pool: int = 50000,
    random_state: int | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fetch row-aligned sign3 vectors for compounds shared by two spaces.

    Two sign3 spaces each cover essentially the whole ~1.2M-compound CC
    universe, so the shared key set is near-total. Materializing it in full
    for both spaces costs roughly ``2 x 1.2e6 x 128 x 4 bytes`` = **1.2 GB**
    of resident memory, which is almost certainly what produced the
    truncated, zero-byte figures seen in earlier runs of this module: they
    appeared at precisely the point where both arrays were live.

    Nothing downstream needs the full set. The recapitulation test samples
    ``n_random`` (2,500) compounds per repetition and the background cutoff
    samples 10,000 pairs, so a random pool of ``max_pool`` shared compounds
    is statistically equivalent and roughly 25x cheaper. The pool is drawn
    once and shared by both spaces so the two vector arrays stay aligned.

    Parameters
    ----------
    sign3_a, sign3_b : chemicalchecker.core.signature_data.DataSignature
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
    shared = sorted(set(sign3_a.keys) & set(sign3_b.keys))
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
            "test (set --max-pool 0 to use all, at ~1.2 GB peak memory).",
            n_shared,
            max_pool,
        )
    else:
        logger.info("Using all %d shared compounds for the recapitulation test.", n_shared)

    # DataSignature.__getitem__ only supports integer fancy indexing
    # (internally slice(min(key), max(key)+1)), which silently breaks for a
    # list of InChIKey strings -- get_vectors() is the library's own
    # fetch-rows-by-key method and handles this correctly. It returns
    # (sorted_keys_found, vectors); both calls query the same key set, so
    # the returned orders match and the arrays are already row-aligned.
    inks_a, vec_a = sign3_a.get_vectors(pool)
    inks_b, vec_b = sign3_b.get_vectors(pool)
    if vec_a is None or vec_b is None:
        raise ValueError("get_vectors() returned no rows for the shared compound set.")
    if not np.array_equal(inks_a, inks_b):
        raise RuntimeError(
            "Key order mismatch between the two spaces' get_vectors() results -- "
            "this should not happen for an identical input key set; inspect "
            "get_vectors() behavior in your chemicalchecker version."
        )
    return list(inks_a), vec_a, vec_b


def shared_key_recapitulation(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    n_shared_compounds: int,
    pval: float = 0.01,
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
    vec_a, vec_b : numpy.ndarray
        Row-aligned sign3 vectors (see :func:`get_shared_vectors`).
    n_shared_compounds : int
        Total number of shared compounds, recorded for the report. This is
        the full intersection size, which may exceed ``len(vec_a)`` when a
        pool subsample was taken.
    pval : float, default 0.01
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
        vec_a, vec_b, pval=pval, n_random=n_random,
        n_subsamples=n_subsamples, random_state=random_state,
    )
    b_by_a = cosine_nn_recapitulation_auroc(
        vec_b, vec_a, pval=pval, n_random=n_random,
        n_subsamples=n_subsamples, random_state=random_state,
    )
    return {
        "n_shared_compounds": int(n_shared_compounds),
        "n_pool_compounds": int(vec_a.shape[0]),
        "pval": pval,
        "a_recap_by_b": a_by_b,
        "b_recap_by_a": b_by_a,
    }


def recapitulation_roc_band(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    pval: float,
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
    pval : float
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
    cutoff, _ = background_distance_cutoff(ref_vectors, pval=pval, random_state=random_state)
    aurocs, tprs, pos_rates = _recapitulation_subsamples(
        ref_vectors, query_vectors, cutoff, n_random, n_subsamples, random_state, fpr_grid
    )
    tprs_arr = np.vstack(tprs)
    return {
        "mean_tpr": tprs_arr.mean(axis=0),
        "std_tpr": tprs_arr.std(axis=0),
        "auroc": float(np.mean(aurocs)),
        "std": float(np.std(aurocs)),
        "cutoff": cutoff,
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
        CC dataset code, e.g. ``"D6.002"``.
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
                "(cosine distance undefined).", dataset_code, st, int(zero.sum()),
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
    pval: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Within-space signature recovery across signature types (Fig. 3b).

    For each ``(lower, higher)`` pair, nearest-neighbour compound pairs are
    defined at the *lower* signature type using the background-distribution
    cutoff at ``pval``, and the *higher* type is scored by AUROC on
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
        CC dataset code, e.g. ``"D6.002"``.
    label : str
        Display label for the space, carried into the returned frame.
    pairs : tuple of (str, str), default SIGNATURE_PAIRS
        ``(lower, higher)`` signature-type combinations to evaluate.
    pval : float, default 0.01
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
            dataset_code, n, n_random, draw,
        )

    rows = []
    for lower, higher in pairs:
        try:
            res = cosine_nn_recapitulation_auroc(
                vectors[lower], vectors[higher], pval=pval, n_random=draw,
                n_subsamples=n_subsamples, random_state=random_state,
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
            dataset_code, lower, higher, res["auroc"], res["std"], n,
        )
    return pd.DataFrame(rows)
