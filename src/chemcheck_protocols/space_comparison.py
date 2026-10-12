"""Comparison of spaces on the compounds they share, from a ``ComparisonConfig``.

Every comparison has two sides (a signature type of a CC dataset, or a matrix file) and is scored
by neighbour recapitulation (Comajuncosa-Creus et al. 2025, Fig. 3 and Extended Data Fig. 8):
pairs of compounds that are nearest neighbours on one side are the positives, and the other side
is scored on ranking them. The scoring is in :mod:`chemcheck_protocols.evaluation`; this module
chooses the compounds, loads the vectors, adds a shuffled null and a distance correlation, and
writes long tables that plotting code can read without touching any signature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .config import Comparison, ComparisonConfig, DataSource, Side, Space
from .data_loaders import load_key_list, load_wide_matrix
from .evaluation import (
    cosine_nn_recapitulation_auroc,
    recapitulation_draw_size,
    recapitulation_roc_band,
)
from .heldout_evaluation import SignatureOpener

logger = logging.getLogger(__name__)

MIN_COMPOUNDS = 50
MEASURED_STAGES = (
    "sign0",
    "sign1",
    "sign2",
)  # the stages that hold only the input compounds
FPR_GRID = np.linspace(0, 1, 101)
DISTANCE_SAMPLE = 1000  # compounds whose pairwise distances are correlated
FULL_MATRIX_LIMIT = (
    3000  # up to this many compounds the null reuses full distance matrices
)
DIRECTIONS = ("a_neighbours_by_b", "b_neighbours_by_a")
COMPARISON_COLUMNS = [
    "comparison",
    "a_space",
    "a_signature",
    "b_space",
    "b_signature",
    "metric",
    "direction",
    "p_value",
    "value",
    "std",
    "null_value",
    "null_std",
    "mean_positive_rate",
    "n_compounds",
]
CURVE_COLUMNS = ["comparison", "direction", "p_value", "fpr", "tpr_mean", "tpr_std"]


class SpaceStore:
    """Opens each signature or matrix once, and answers which compounds and vectors it has."""

    def __init__(self, config: ComparisonConfig, open_signature: SignatureOpener):
        self._spaces: dict[str, Space] = {space.name: space for space in config.spaces}
        self._open_signature = open_signature
        self._signatures: dict[tuple[str, str], object] = {}
        self._matrices: dict[str, pd.DataFrame] = {}

    def _signature(self, space: Space, signature: str):
        key = (space.name, signature)
        if key not in self._signatures:
            self._signatures[key] = self._open_signature(
                space.cc_root, space.dataset_code, signature
            )
        return self._signatures[key]

    def _matrix(self, space: Space) -> pd.DataFrame:
        if space.name not in self._matrices:
            source = DataSource(
                format="wide_matrix", path=space.matrix, separator=space.separator
            )
            matrix = load_wide_matrix(source).groupby(level=0).mean().fillna(0.0)
            self._matrices[space.name] = matrix
        return self._matrices[space.name]

    def keys(self, side: Side) -> set[str]:
        """Compounds the side has."""
        space = self._spaces[side.space]
        if not space.is_cc:
            return set(map(str, self._matrix(space).index))
        return set(map(str, self._signature(space, side.signature).keys))

    def measured_keys(self, side: Side) -> set[str]:
        """
        Compounds with real data: those in sign0, sign1 and sign2 of a CC dataset (sign1 and sign2
        drop the input compounds the sanitizer filtered out), or the rows of a matrix.
        """
        space = self._spaces[side.space]
        if not space.is_cc:
            return self.keys(side)
        return set.intersection(
            *(
                set(map(str, self._signature(space, stage).keys))
                for stage in MEASURED_STAGES
            )
        )

    def vectors(self, side: Side, keys: list[str]) -> tuple[list[str], np.ndarray]:
        """The requested compounds the side has, and their vectors, row-aligned."""
        space = self._spaces[side.space]
        if not space.is_cc:
            matrix = self._matrix(space)
            found = [key for key in keys if key in matrix.index]
            return found, matrix.loc[found].to_numpy(dtype=np.float64)
        found, vectors = self._signature(space, side.signature).get_vectors(keys)
        return [str(key) for key in found], np.asarray(vectors, dtype=np.float64)


def choose_compounds(
    store: SpaceStore, comparison: Comparison, config: ComparisonConfig
) -> list[str]:
    """
    The compounds a comparison is computed on, by ``config.compounds`` (see
    :class:`~chemcheck_protocols.config.ComparisonConfig`), at most ``config.max_compounds``.
    """
    sides = (comparison.a, comparison.b)
    keys = set.intersection(*(store.keys(side) for side in sides))
    if config.compounds == "measured":
        keys &= set.intersection(*(store.measured_keys(side) for side in sides))
    elif config.compounds == "from_file":
        keys &= set(load_key_list(config.compounds_file))
    chosen = sorted(keys)
    if len(chosen) > config.max_compounds:
        rng = np.random.default_rng(config.seed)
        chosen = sorted(rng.choice(chosen, size=config.max_compounds, replace=False))
        logger.info(
            "%s: drew %d of %d compounds", comparison.label, len(chosen), len(keys)
        )
    return chosen


def shared_vectors(
    store: SpaceStore, comparison: Comparison, keys: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Row-aligned vectors of ``keys`` on both sides, without compounds whose vector is zero or not
    finite on either side (cosine distance is undefined for them).

    Raises
    ------
    ValueError
        If fewer than ``MIN_COMPOUNDS`` compounds remain.
    """
    found_a, vectors_a = store.vectors(comparison.a, keys)
    found_b, vectors_b = store.vectors(comparison.b, keys)
    common = sorted(set(found_a) & set(found_b))
    vectors_a = vectors_a[[found_a.index(key) for key in common]]
    vectors_b = vectors_b[[found_b.index(key) for key in common]]
    good = np.ones(len(common), dtype=bool)
    for vectors in (vectors_a, vectors_b):
        good &= np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) > 0)
    if not good.all():
        logger.warning(
            "%s: dropping %d compound(s) with a zero or non-finite vector",
            comparison.label,
            int((~good).sum()),
        )
    kept = [key for key, keep in zip(common, good) if keep]
    if len(kept) < MIN_COMPOUNDS:
        raise ValueError(
            f"{comparison.label}: only {len(kept)} compounds to compare "
            f"(see the compounds setting); at least {MIN_COMPOUNDS} are needed"
        )
    return kept, vectors_a[good], vectors_b[good]


def shuffled_null(
    reference: np.ndarray,
    query: np.ndarray,
    n_permutations: int,
    draw: int,
    p_value_cutoff: float,
    seed: int,
    cutoff: float | None = None,
    distances: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float]:
    """
    Mean and std of the recapitulation AUROC when the compounds of ``query`` are shuffled.

    With ``distances`` (the square cosine-distance matrices of ``reference`` and ``query``) and
    the neighbour ``cutoff`` of the reference, a shuffle only reorders the distances, which is
    much cheaper than recomputing them in the full feature space; without them every shuffle is
    scored from the vectors.
    """
    rng = np.random.default_rng(seed)
    n = len(reference)
    upper = np.triu_indices(draw, k=1)
    aurocs = []
    for repetition in range(n_permutations):
        if distances is None:
            try:
                result = cosine_nn_recapitulation_auroc(
                    reference,
                    query[rng.permutation(n)],
                    p_value_cutoff=p_value_cutoff,
                    n_random=draw,
                    n_subsamples=1,
                    random_state=seed + repetition,
                )
            except RuntimeError:
                continue  # degenerate draw (no or all pairs within the cutoff)
            aurocs.append(result["auroc"])
            continue
        permutation = rng.permutation(n)
        chosen = rng.choice(n, size=draw, replace=False)
        positives = distances[0][np.ix_(chosen, chosen)][upper] <= cutoff
        if positives.all() or not positives.any():
            continue
        shuffled = permutation[chosen]
        scores = -distances[1][np.ix_(shuffled, shuffled)][upper]
        aurocs.append(float(roc_auc_score(positives, scores)))
    if not aurocs:
        return float("nan"), float("nan")
    return float(np.mean(aurocs)), float(np.std(aurocs))


def distance_correlation(
    vectors_a: np.ndarray, vectors_b: np.ndarray, seed: int
) -> float:
    """Spearman correlation of the pairwise cosine distances of the two sides."""
    rng = np.random.default_rng(seed)
    n = len(vectors_a)
    sample = rng.choice(n, size=min(DISTANCE_SAMPLE, n), replace=False)
    return float(
        spearmanr(
            pdist(vectors_a[sample], "cosine"), pdist(vectors_b[sample], "cosine")
        ).statistic
    )


def compare_vectors(
    comparison: Comparison,
    vectors_a: np.ndarray,
    vectors_b: np.ndarray,
    config: ComparisonConfig,
) -> tuple[list[dict], list[dict]]:
    """Table rows and ROC-curve rows of one comparison, from row-aligned vectors."""
    n = len(vectors_a)
    draw = recapitulation_draw_size(n, config.n_random, config.n_subsamples)
    identity = {
        "comparison": comparison.label,
        "a_space": comparison.a.space,
        "a_signature": comparison.a.signature,
        "b_space": comparison.b.space,
        "b_signature": comparison.b.signature,
        "n_compounds": n,
    }
    matrices = (
        (squareform(pdist(vectors_a, "cosine")), squareform(pdist(vectors_b, "cosine")))
        if n <= FULL_MATRIX_LIMIT
        else None
    )
    rows, curves = [], []
    for direction, reference, query in zip(
        DIRECTIONS, (vectors_a, vectors_b), (vectors_b, vectors_a)
    ):
        for p_value in config.p_value_cutoffs:
            try:
                band = recapitulation_roc_band(
                    reference,
                    query,
                    p_value,
                    draw,
                    config.n_subsamples,
                    config.seed,
                    FPR_GRID,
                )
            except RuntimeError as error:
                logger.warning(
                    "%s %s p=%g: skipped: %s",
                    comparison.label,
                    direction,
                    p_value,
                    error,
                )
                continue
            null_value, null_std = shuffled_null(
                reference,
                query,
                config.n_permutations,
                draw,
                p_value,
                config.seed,
                cutoff=band["cutoff"],
                distances=None
                if matrices is None
                else (matrices if direction == DIRECTIONS[0] else matrices[::-1]),
            )
            rows.append(
                {
                    **identity,
                    "metric": "recapitulation_auroc",
                    "direction": direction,
                    "p_value": p_value,
                    "value": band["auroc"],
                    "std": band["std"],
                    "null_value": null_value,
                    "null_std": null_std,
                    "mean_positive_rate": band["mean_positive_rate"],
                }
            )
            curves += [
                {
                    "comparison": comparison.label,
                    "direction": direction,
                    "p_value": p_value,
                    "fpr": fpr,
                    "tpr_mean": mean,
                    "tpr_std": std,
                }
                for fpr, mean, std in zip(FPR_GRID, band["mean_tpr"], band["std_tpr"])
            ]
            logger.info(
                "%s %s p=%g: AUROC=%.3f +/- %.3f (shuffled %.3f, n=%d)",
                comparison.label,
                direction,
                p_value,
                band["auroc"],
                band["std"],
                null_value,
                n,
            )
    rows.append(
        {
            **identity,
            "metric": "distance_spearman",
            "value": distance_correlation(vectors_a, vectors_b, config.seed),
        }
    )
    return rows, curves


def run_comparison(
    config: ComparisonConfig, open_signature: SignatureOpener
) -> dict[str, pd.DataFrame]:
    """
    Run every comparison of the configuration.

    Parameters
    ----------
    config : ComparisonConfig
        The validated comparison.
    open_signature : SignatureOpener
        How signatures are opened (``cc_signature_opener()`` in production).

    Returns
    -------
    dict of str to pandas.DataFrame
        ``comparisons``: one row per comparison, metric (``recapitulation_auroc`` per
        direction and p-value, with its shuffled null; ``distance_spearman``) and
        ``n_compounds``; ``curves``: the mean ROC curve and its spread for each
        recapitulation row.

    Raises
    ------
    ValueError
        If a comparison has too few compounds.
    """
    store = SpaceStore(config, open_signature)
    rows, curves = [], []
    comparisons = config.comparisons()
    for number, comparison in enumerate(comparisons, start=1):
        logger.info("Comparison %d/%d: %s", number, len(comparisons), comparison.label)
        keys = choose_compounds(store, comparison, config)
        _, vectors_a, vectors_b = shared_vectors(store, comparison, keys)
        new_rows, new_curves = compare_vectors(comparison, vectors_a, vectors_b, config)
        rows += new_rows
        curves += new_curves
    return {
        "comparisons": pd.DataFrame(rows, columns=COMPARISON_COLUMNS),
        "curves": pd.DataFrame(curves, columns=CURVE_COLUMNS),
    }


def write_comparison(
    tables: dict[str, pd.DataFrame], config: ComparisonConfig
) -> list[Path]:
    """
    Write the result tables and a record of the run next to the output prefix.

    ``<output>_comparisons.csv`` and ``<output>_curves.csv`` are the tables of
    :func:`run_comparison`; ``<output>_run.yaml`` holds the settings used and the list of
    comparisons, so a result can be read without the original configuration file.
    """
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = output.parent / f"{output.name}_{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    record = config.model_dump(
        mode="json", by_alias=True, exclude={"comparisons_listed"}
    )
    record["comparisons"] = [comparison.label for comparison in config.comparisons()]
    run_path = output.parent / f"{output.name}_run.yaml"
    run_path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    paths.append(run_path)
    logger.info("Wrote %s", ", ".join(str(path) for path in paths))
    return paths
