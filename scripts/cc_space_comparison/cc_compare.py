"""Core logic for comparing two Chemical Checker bioactivity spaces.

This module implements the space-vs-space validation analyses described in
Comajuncosa-Creus et al. (Nat. Protoc. 2025) -- specifically the
shared-compound nearest-neighbor recapitulation test (their Fig. 3a/b and
Extended Data Fig. 8g,h) and side-by-side comparisons of the diagnosis
artifacts that ``signature.diagnosis().canvas()`` already writes to disk
(``diags/<run_name>_sign3/*.pkl`` and ``stats/*.tsv`` / ``validation_stats.json``).

All functions here are pure/stateless with respect to a fitted CC instance:
callers pass in already-instantiated ``ChemicalChecker`` objects and dataset
codes, and this module handles locating diagnosis artifacts on disk,
recomputing the shared-key restricted metrics, and producing comparison
tables/figures. No signature fitting happens here -- see ``cc_pipeline.py``
(or your existing pipeline) for that.

Notes
-----
The exact keys/columns inside ``across_roc.pkl``, ``confidences.pkl``, etc.
depend on the installed ``chemicalchecker`` version. The loaders below make a
best-effort, documented assumption about each artifact's shape and raise a
clear error (rather than silently mis-parsing) if that assumption does not
hold -- inspect the offending ``.pkl`` once with ``pickle.load`` and adjust
the relevant ``_coerce_*`` helper if your version differs.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class SpaceSpec:
    """Identifies one bioactivity space to compare, within a CC instance.

    Parameters
    ----------
    label : str
        Human-readable name used in tables and plot legends (e.g.
        ``"DeepCoverMoA only"``).
    dataset_code : str
        CC dataset code as registered in the local CC instance (e.g.
        ``"D6.002"``).
    """

    label: str
    dataset_code: str


def _dataset_signature_dir(local_cc_dir: Path, dataset_code: str, sign_type: str) -> Path:
    """Build the on-disk path to a dataset's signature directory.

    Follows the CC folder convention
    ``full/<level>/<coordinate>/<dataset_code>/<sign_type>``, e.g.
    ``local_CC_D6/full/D/D6/D6.002/sign3``.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    sign_type : str
        Signature type, e.g. ``"sign0"``, ``"sign1"``, ``"sign2"``, ``"sign3"``.

    Returns
    -------
    Path
        Path to the dataset's signature directory.
    """
    level = dataset_code[0]
    coordinate = dataset_code[:2]
    return local_cc_dir / "full" / level / coordinate / dataset_code / sign_type


def _find_diag_run_dir(sign_dir: Path) -> Path:
    """Locate the single ``diags/<run_name>_sign3``-style directory.

    ``signature.diagnosis().canvas()`` writes its artifacts under a
    subdirectory named after the local CC instance (e.g.
    ``diags/local_CC_D6_sign3``). The exact run name depends on how the
    instance directory was named, so this locates it by globbing rather
    than hardcoding it.

    Parameters
    ----------
    sign_dir : Path
        Path to a dataset's signature directory (e.g. ``.../D6.002/sign3``).

    Returns
    -------
    Path
        Path to the single matching diagnosis run directory.

    Raises
    ------
    FileNotFoundError
        If no diagnosis run directory is found under ``sign_dir / "diags"``.
    RuntimeError
        If more than one candidate directory is found, since the choice
        would then be ambiguous.
    """
    diags_dir = sign_dir / "diags"
    candidates = sorted(diags_dir.glob("*_sign3"))
    if not candidates:
        raise FileNotFoundError(
            f"No diagnosis run directory found under {diags_dir}. "
            "Has diagnosis().canvas() been run for this dataset?"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple diagnosis run directories found under {diags_dir}: "
            f"{[c.name for c in candidates]}. Disambiguate manually."
        )
    return candidates[0]


def load_diag_artifact(
    local_cc_dir: Path, dataset_code: str, name: str, sign_type: str = "sign3"
) -> Any:
    """Load one pickled diagnosis artifact for a dataset.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    name : str
        Artifact name without extension, e.g. ``"confidences"``,
        ``"across_roc"``, ``"outliers"``, ``"redundancy"``.
    sign_type : str, default "sign3"
        Signature type the diagnosis was run on.

    Returns
    -------
    Any
        The unpickled artifact (type depends on ``name``).

    Raises
    ------
    FileNotFoundError
        If the artifact file does not exist.
    """
    sign_dir = _dataset_signature_dir(local_cc_dir, dataset_code, sign_type)
    run_dir = _find_diag_run_dir(sign_dir)
    artifact_path = run_dir / f"{name}.pkl"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Diagnosis artifact not found: {artifact_path}")
    with open(artifact_path, "rb") as fh:
        return pickle.load(fh)


def load_validation_table(
    local_cc_dir: Path, dataset_code: str, kind: str, sign_type: str = "sign3"
) -> pd.DataFrame:
    """Load the MoA/ATC AUROC validation table written by ``diagnosis()``.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    kind : str
        Either ``"moa"`` or ``"atc"``.
    sign_type : str, default "sign3"
        Signature type the diagnosis was run on.

    Returns
    -------
    pandas.DataFrame
        Contents of ``stats/<kind>_<sign_type>_auc_validation.tsv``.
    """
    if kind not in {"moa", "atc"}:
        raise ValueError(f"kind must be 'moa' or 'atc', got {kind!r}")
    stats_dir = _dataset_signature_dir(local_cc_dir, dataset_code, sign_type) / "stats"
    path = stats_dir / f"{kind}_{sign_type}_auc_validation.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Validation table not found: {path}")
    return pd.read_csv(path, sep="\t")


def load_validation_stats(
    local_cc_dir: Path, dataset_code: str, sign_type: str = "sign3"
) -> dict[str, Any]:
    """Load ``stats/validation_stats.json`` for a dataset.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    sign_type : str, default "sign3"
        Signature type the diagnosis was run on.

    Returns
    -------
    dict
        Parsed contents of ``validation_stats.json``.
    """
    stats_dir = _dataset_signature_dir(local_cc_dir, dataset_code, sign_type) / "stats"
    path = stats_dir / "validation_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"validation_stats.json not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_1d_array(raw: Any, name: str) -> np.ndarray:
    """Best-effort coercion of a diagnosis artifact into a flat 1D array.

    Different ``chemicalchecker`` versions store per-compound diagnosis
    values (``confidences.pkl``, ``outliers.pkl``, etc.) under slightly
    different shapes: a bare ``numpy.ndarray``, a ``pandas.Series``, or a
    ``dict`` wrapping the array under a key such as ``"V"`` (the CC
    convention for a signature's main data matrix) or the artifact's own
    name. This function tries each in turn rather than assuming one shape.

    Parameters
    ----------
    raw : Any
        Unpickled artifact contents.
    name : str
        Artifact name (e.g. ``"confidences"``), used both as a candidate
        dict key and in the error message if coercion fails.

    Returns
    -------
    numpy.ndarray
        A flat 1D array of values.

    Raises
    ------
    TypeError
        If ``raw`` does not match any recognized shape.
    """
    if isinstance(raw, np.ndarray):
        return raw.ravel()
    if isinstance(raw, pd.Series):
        return raw.to_numpy().ravel()
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] == 1:
            return raw.iloc[:, 0].to_numpy().ravel()
        raise TypeError(
            f"'{name}' DataFrame has {raw.shape[1]} columns; expected a single "
            f"value column. Columns: {list(raw.columns)}."
        )
    if isinstance(raw, dict):
        for key in ("V", "values", "value", name, name.rstrip("s"), "scores", "score"):
            if key in raw:
                return _coerce_1d_array(raw[key], name)
        values = list(raw.values())
        if values and all(isinstance(v, (int, float, np.floating, np.integer)) for v in values):
            # Looks like a {compound_key: value} mapping.
            return np.asarray(values, dtype=float)
        raise TypeError(
            f"Could not coerce '{name}' dict payload into a 1D array "
            f"(top-level keys: {list(raw.keys())[:10]}). Inspect it manually "
            f"(pickle.load) and extend _coerce_1d_array()."
        )
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw).ravel()
    raise TypeError(f"Unrecognized '{name}' payload type: {type(raw)!r}.")


def _coerce_across_roc(raw: Any) -> pd.DataFrame:
    """Best-effort coercion of an ``across_roc.pkl`` payload into a tidy frame.

    Handles the two shapes commonly produced by ``diagnosis()``: a
    ``pandas.DataFrame`` indexed by CC dataset code, or a ``dict`` mapping
    dataset code to AUROC.

    Parameters
    ----------
    raw : Any
        Unpickled contents of ``across_roc.pkl``.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``cc_space`` and ``auroc``.

    Raises
    ------
    TypeError
        If ``raw`` matches neither expected shape.
    """
    if isinstance(raw, pd.DataFrame):
        df = raw.reset_index()
        df = df.rename(columns={df.columns[0]: "cc_space", df.columns[1]: "auroc"})
        return df[["cc_space", "auroc"]]
    if isinstance(raw, dict):
        return pd.DataFrame({"cc_space": list(raw.keys()), "auroc": list(raw.values())})
    raise TypeError(
        f"Unrecognized across_roc.pkl payload type: {type(raw)!r}. "
        "Inspect it manually (pickle.load) and extend _coerce_across_roc()."
    )


def compare_across_roc(
    local_cc_dir_a: Path, dataset_a: str, local_cc_dir_b: Path, dataset_b: str
) -> pd.DataFrame:
    """Compare per-CC-space nearest-neighbor recapitulation AUROCs.

    This reproduces the "ROC across CC" diagnosis panel for both spaces and
    merges them on the shared CC dataset code, so each of the 25 CC spaces
    becomes one row with an AUROC from each of the two compared spaces.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B.
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.

    Returns
    -------
    pandas.DataFrame
        Columns: ``cc_space``, ``auroc_a``, ``auroc_b``.
    """
    raw_a = load_diag_artifact(local_cc_dir_a, dataset_a, "across_roc")
    raw_b = load_diag_artifact(local_cc_dir_b, dataset_b, "across_roc")
    df_a = _coerce_across_roc(raw_a)
    df_b = _coerce_across_roc(raw_b)
    return df_a.merge(df_b, on="cc_space", suffixes=("_a", "_b"))


def compare_confidence(
    local_cc_dir_a: Path, dataset_a: str, local_cc_dir_b: Path, dataset_b: str
) -> pd.DataFrame:
    """Stack confidence-score distributions from two spaces into one frame.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B.
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.

    Returns
    -------
    pandas.DataFrame
        Columns: ``space``, ``confidence``.
    """
    conf_a = _coerce_1d_array(load_diag_artifact(local_cc_dir_a, dataset_a, "confidences"), "confidences")
    conf_b = _coerce_1d_array(load_diag_artifact(local_cc_dir_b, dataset_b, "confidences"), "confidences")
    return pd.DataFrame(
        {
            "space": [dataset_a] * len(conf_a) + [dataset_b] * len(conf_b),
            "confidence": np.concatenate([conf_a, conf_b]),
        }
    )


def build_summary_table(specs: list[SpaceSpec], local_cc_dirs: dict[str, Path]) -> pd.DataFrame:
    """One row per space with headline validation numbers pulled from disk.

    Collects the same numbers shown in the diagnosis canvas titles
    (MoA/ATC AUROC, mean confidence, outlier proportion) programmatically,
    instead of reading them off the figure.

    Parameters
    ----------
    specs : list of SpaceSpec
        Spaces to summarize.
    local_cc_dirs : dict
        Maps each ``SpaceSpec.label`` to its local CC instance directory.

    Returns
    -------
    pandas.DataFrame
        One row per space.
    """
    rows = []
    for spec in specs:
        cc_dir = local_cc_dirs[spec.label]
        stats = load_validation_stats(cc_dir, spec.dataset_code)
        confidences = _coerce_1d_array(
            load_diag_artifact(cc_dir, spec.dataset_code, "confidences"), "confidences"
        )
        outliers = _coerce_1d_array(
            load_diag_artifact(cc_dir, spec.dataset_code, "outliers"), "outliers"
        )
        row: dict[str, Any] = {
            "space": spec.label,
            "dataset_code": spec.dataset_code,
            "n_keys": int(len(confidences)),
            "mean_confidence": float(np.mean(confidences)),
            "pct_outliers": float(np.mean(outliers > 0) * 100),
        }
        row.update({f"validation_{k}": v for k, v in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def cosine_nn_recapitulation_auroc(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    pval: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> tuple[float, float]:
    """AUROC for recapitulating reference-space nearest neighbors in a query space.

    Implements the recapitulation test described in Comajuncosa-Creus et al.
    (Nat. Protoc. 2025, Fig. 3a): compound pairs that are nearest neighbors
    in ``ref_vectors`` (cosine distance below the ``pval``-th percentile of
    the pairwise distance distribution) are treated as positives; the
    ``query_vectors`` cosine distance for the same pairs is used as the
    ranking score. Repeated over ``n_subsamples`` random draws of
    ``n_random`` rows and averaged.

    Parameters
    ----------
    ref_vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors defining ground-truth nearest neighbors.
    query_vectors : numpy.ndarray, shape (n_compounds, n_features)
        Vectors being evaluated for their ability to recapitulate those
        neighbors. Must be row-aligned with ``ref_vectors``.
    pval : float, default 0.01
        Percentile of the pairwise cosine-distance distribution used as the
        nearest-neighbor cutoff.
    n_random : int, default 2500
        Number of compounds subsampled per repetition.
    n_subsamples : int, default 5
        Number of repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.

    Returns
    -------
    tuple of (float, float)
        Mean and standard deviation of the AUROC across subsamples.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate (single-class) label set.
    """
    rng = np.random.default_rng(random_state)
    n = ref_vectors.shape[0]
    n_random = min(n_random, n)
    aurocs = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=n_random, replace=False)
        d_ref = squareform(pdist(ref_vectors[idx], metric="cosine"))
        d_query = squareform(pdist(query_vectors[idx], metric="cosine"))
        iu = np.triu_indices_from(d_ref, k=1)
        cutoff = np.quantile(d_ref[iu], pval)
        y_true = (d_ref[iu] <= cutoff).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("Degenerate NN cutoff at pval=%.4f; skipping this subsample.", pval)
            continue
        y_score = -d_query[iu]
        aurocs.append(roc_auc_score(y_true, y_score))
    if not aurocs:
        raise RuntimeError(
            "All subsamples were degenerate (no valid NN cutoff); try a larger "
            "n_random or a different pval."
        )
    return float(np.mean(aurocs)), float(np.std(aurocs))


def get_shared_vectors(sign3_a, sign3_b) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fetch row-aligned sign3 vectors for compounds shared by two spaces.

    Parameters
    ----------
    sign3_a, sign3_b : chemicalchecker.core.signature_data.DataSignature
        Fitted ``sign3`` signature objects for the two spaces being compared.

    Returns
    -------
    tuple of (list of str, numpy.ndarray, numpy.ndarray)
        The shared keys (sorted), and their vectors from space A and space B
        respectively, row-aligned to each other.

    Raises
    ------
    ValueError
        If fewer than 50 compounds are shared between the two spaces, or if
        ``get_vectors()`` returns no rows.
    RuntimeError
        If the two spaces' ``get_vectors()`` calls return keys in different
        order (should not happen for an identical input key set).
    """
    keys_a = set(sign3_a.keys)
    keys_b = set(sign3_b.keys)
    shared = sorted(keys_a & keys_b)
    if len(shared) < 50:
        raise ValueError(
            f"Only {len(shared)} compounds shared between the two spaces; "
            "too few for a reliable recapitulation estimate."
        )
    logger.info("Restricting recapitulation test to %d shared compounds.", len(shared))

    # DataSignature.__getitem__ only supports list-of-int fancy indexing
    # (internally it does slice(min(key), max(key)+1), which breaks for a
    # list of InChIKey strings) -- get_vectors() is the library's own method
    # for fetching rows by key and handles this correctly. It returns
    # (sorted_keys_found, vectors); since both calls query the exact same
    # `shared` key set, the returned key order is guaranteed identical
    # between the two spaces (both sort the same key set the same way), so
    # the two vector arrays are already row-aligned.
    inks_a, vec_a = sign3_a.get_vectors(shared)
    inks_b, vec_b = sign3_b.get_vectors(shared)
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
    """Bidirectional NN-recapitulation AUROC restricted to shared compounds.

    Restricting to compounds present in both spaces isolates "did adding
    data hurt the signatures of the original compounds" from "the newly
    added compounds are just harder to signaturize" -- comparing whole-space
    AUROCs (as printed in the diagnosis canvas titles) conflates the two,
    since the two spaces cover different, differently sized compound sets.

    Parameters
    ----------
    vec_a, vec_b : numpy.ndarray
        Row-aligned sign3 vectors for the shared compounds (see
        :func:`get_shared_vectors`).
    n_shared_compounds : int
        Number of shared compounds (``len(vec_a)``), recorded for the report.
    pval : float, default 0.01
        Nearest-neighbor cosine-distance percentile cutoff (see
        :func:`cosine_nn_recapitulation_auroc`).
    n_random : int, default 2500
        Number of shared compounds subsampled per repetition.
    n_subsamples : int, default 5
        Number of repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        Keys: ``n_shared_compounds``, ``auroc_a_recap_by_b``,
        ``std_a_recap_by_b``, ``auroc_b_recap_by_a``, ``std_b_recap_by_a``.
    """
    auroc_a_by_b, std_a_by_b = cosine_nn_recapitulation_auroc(
        vec_a, vec_b, pval=pval, n_random=n_random, n_subsamples=n_subsamples, random_state=random_state
    )
    auroc_b_by_a, std_b_by_a = cosine_nn_recapitulation_auroc(
        vec_b, vec_a, pval=pval, n_random=n_random, n_subsamples=n_subsamples, random_state=random_state
    )
    return {
        "n_shared_compounds": n_shared_compounds,
        "auroc_a_recap_by_b": auroc_a_by_b,
        "std_a_recap_by_b": std_a_by_b,
        "auroc_b_recap_by_a": auroc_b_by_a,
        "std_b_recap_by_a": std_b_by_a,
    }


def _roc_curve_with_band(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    pval: float,
    n_random: int,
    n_subsamples: int,
    random_state: int | None,
    fpr_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute a mean +/- std ROC curve across subsamples, on a common FPR grid.

    Same NN-definition methodology as :func:`cosine_nn_recapitulation_auroc`,
    but keeps the full ROC curve (interpolated onto ``fpr_grid``) from each
    subsample instead of collapsing straight to an AUROC scalar -- this is
    what reproduces the shaded-uncertainty-band ROC plots in Fig. 3g,h of
    Comajuncosa-Creus et al., rather than just the number in Fig. 3b.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors; NN pairs are defined on ``ref_vectors`` and
        recapitulation is scored against ``query_vectors``.
    pval : float
        Nearest-neighbor cosine-distance percentile cutoff.
    n_random : int
        Number of compounds subsampled per repetition.
    n_subsamples : int
        Number of repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.
    fpr_grid : numpy.ndarray
        Common FPR grid (e.g. ``np.linspace(0, 1, 100)``) that each
        subsample's ROC curve is interpolated onto before averaging.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, float, float)
        ``(mean_tpr, std_tpr, mean_auroc, std_auroc)``, all evaluated on
        ``fpr_grid``.

    Raises
    ------
    RuntimeError
        If every subsample produced a degenerate (single-class) label set.
    """
    rng = np.random.default_rng(random_state)
    n = ref_vectors.shape[0]
    n_random = min(n_random, n)
    tprs, aurocs = [], []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=n_random, replace=False)
        d_ref = squareform(pdist(ref_vectors[idx], metric="cosine"))
        d_query = squareform(pdist(query_vectors[idx], metric="cosine"))
        iu = np.triu_indices_from(d_ref, k=1)
        cutoff = np.quantile(d_ref[iu], pval)
        y_true = (d_ref[iu] <= cutoff).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("Degenerate NN cutoff at pval=%.4f; skipping this subsample.", pval)
            continue
        y_score = -d_query[iu]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aurocs.append(auc(fpr, tpr))
    if not aurocs:
        raise RuntimeError(
            "All subsamples were degenerate (no valid NN cutoff); try a larger "
            "n_random or a different pval."
        )
    tprs_arr = np.vstack(tprs)
    return tprs_arr.mean(axis=0), tprs_arr.std(axis=0), float(np.mean(aurocs)), float(np.std(aurocs))


def plot_recapitulation_roc(
    ref_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ref_label: str,
    query_label: str,
    output_path: Path,
    pvals: tuple[float, ...] = (0.01, 0.001),
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
) -> None:
    """ROC-with-uncertainty-band plot for NN recapitulation (Fig. 3g,h style).

    For each cutoff in ``pvals``, plots the mean ROC curve (across
    ``n_subsamples`` random draws) with a shaded +/- 1 std band, for how well
    ``query_vectors`` recapitulates nearest neighbors defined at
    ``ref_vectors``.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors (see :func:`get_shared_vectors`).
    ref_label, query_label : str
        Display labels for the reference and query spaces.
    output_path : Path
        Where to save the figure (PNG).
    pvals : tuple of float, default (0.01, 0.001)
        Nearest-neighbor cosine-distance percentile cutoffs to plot, one
        curve each (matches the two cutoffs used in the CC Protocols paper).
    n_random, n_subsamples, random_state
        Forwarded to :func:`_roc_curve_with_band`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr_grid = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(pvals)))
    for pval, color in zip(pvals, colors):
        mean_tpr, std_tpr, mean_auroc, std_auroc = _roc_curve_with_band(
            ref_vectors, query_vectors, pval, n_random, n_subsamples, random_state, fpr_grid
        )
        ax.plot(
            fpr_grid, mean_tpr, color=color,
            label=f"pval:{pval:.1e} - {mean_auroc:.2f}\u00b1{std_auroc:.2f}",
        )
        ax.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Recap. {ref_label} NN by {query_label}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confidence_overlay(df: pd.DataFrame, output_path: Path) -> None:
    """Overlay confidence-score histograms for two spaces and save to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compare_confidence`.
    output_path : Path
        Where to save the figure (PNG).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for space, group in df.groupby("space"):
        ax.hist(group["confidence"], bins=50, alpha=0.5, density=True, label=space)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_across_roc_scatter(df: pd.DataFrame, output_path: Path, label_a: str, label_b: str) -> None:
    """Scatter per-CC-space AUROCs of two compared spaces against each other.

    Points above the y=x diagonal recapitulate better in space B than in
    space A, and vice versa -- useful to spot whether a change affects all
    25 CC spaces uniformly or is localized to specific levels.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compare_across_roc`.
    output_path : Path
        Where to save the figure (PNG).
    label_a, label_b : str
        Display labels for the two compared spaces.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["auroc_a"], df["auroc_b"], s=20)
    lims = [0.4, 1.0]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{label_a} ROC-AUC")
    ax.set_ylabel(f"{label_b} ROC-AUC")
    ax.set_title("Per-CC-space NN recapitulation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _extract_roc_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort extraction of (fpr, tpr) arrays from a validation table.

    Column names are matched case-insensitively against ``"fpr"``/``"tpr"``
    substrings, since the exact header used by different ``chemicalchecker``
    versions is not guaranteed to be literally ``"fpr"``/``"tpr"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_validation_table`.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``(fpr, tpr)`` arrays, in the row order found in ``df``.

    Raises
    ------
    KeyError
        If no column name contains ``"fpr"`` or ``"tpr"``.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    fpr_col = next((orig for low, orig in cols_lower.items() if "fpr" in low), None)
    tpr_col = next((orig for low, orig in cols_lower.items() if "tpr" in low), None)
    if fpr_col is None or tpr_col is None:
        raise KeyError(
            f"Could not find fpr/tpr columns in validation table (columns: "
            f"{list(df.columns)}). Inspect the *_auc_validation.tsv file and "
            "extend _extract_roc_curve()."
        )
    return df[fpr_col].to_numpy(dtype=float), df[tpr_col].to_numpy(dtype=float)


def plot_validation_roc_overlay(
    local_cc_dir_a: Path,
    dataset_a: str,
    label_a: str,
    local_cc_dir_b: Path,
    dataset_b: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Overlay MoA and ATC ROC curves from both spaces' validation tables.

    Reproduces the diagnosis canvas's "MoA (AUROC)" / "ATC (AUROC)" panels
    for two spaces on the same axes, so curve shape -- not just the AUROC
    scalar shown in the canvas title -- can be compared directly.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B.
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.
    label_a, label_b : str
        Display labels for the two compared spaces.
    output_path : Path
        Where to save the figure (PNG).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, kind in zip(axes, ("moa", "atc")):
        for cc_dir, dataset, label in (
            (local_cc_dir_a, dataset_a, label_a),
            (local_cc_dir_b, dataset_b, label_b),
        ):
            df = load_validation_table(cc_dir, dataset, kind)
            fpr, tpr = _extract_roc_curve(df)
            order = np.argsort(fpr)
            ax.plot(fpr[order], tpr[order], label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(kind.upper())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_headline_metrics_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Small-multiple bar charts of headline validation metrics for both spaces.

    Combines the numbers shown in the diagnosis canvas titles (MoA AUROC,
    ATC AUROC, if present under ``validation_*`` columns) with mean
    confidence and outlier proportion. Each metric gets its own subplot with
    its own y-axis, rather than one shared-axis grouped bar chart -- sharing
    an axis breaks badly here because ``validation_stats.json`` also carries
    raw compound counts (e.g. a ``molecules`` field in the ~1e6 range) that
    would otherwise dwarf every 0-1-scale AUROC/confidence bar into
    invisibility. Count-like fields are excluded outright rather than merely
    rescaled, since they aren't validation metrics.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of :func:`build_summary_table`.
    output_path : Path
        Where to save the figure (PNG).

    Raises
    ------
    ValueError
        If no numeric, non-count metric columns are found to plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exclude_name_hints = ("molecule", "n_keys", "keys", "count")
    metric_cols = [
        c
        for c in summary_df.columns
        if (c.startswith("validation_") or c in {"mean_confidence", "pct_outliers"})
        and pd.api.types.is_numeric_dtype(summary_df[c])
        and not any(hint in c.lower() for hint in exclude_name_hints)
    ]
    if not metric_cols:
        raise ValueError(
            "No numeric, non-count metric columns found in summary_df to plot "
            f"(columns available: {list(summary_df.columns)})."
        )

    ncols = min(4, len(metric_cols))
    nrows = int(np.ceil(len(metric_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    spaces = summary_df["space"].tolist()
    for ax, col in zip(axes_flat, metric_cols):
        ax.bar(spaces, summary_df[col].to_numpy(dtype=float))
        ax.set_title(col.replace("validation_", ""), fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    for ax in axes_flat[len(metric_cols):]:
        ax.axis("off")

    fig.suptitle("Headline validation metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def get_sign0_keys(cc, dataset_code: str) -> set[str]:
    """Return the set of compound keys with real (non-inferred) sign0 data.

    Used to define genuine dataset membership, as opposed to sign3 key
    coverage: sign3 is inferred for essentially the entire ~1M-compound CC
    universe regardless of whether a given compound had real data for this
    specific space, so a membership overlay built from sign3 keys would show
    almost nothing (both spaces' sign3 cover nearly the same universe). This
    is the same notion of "compounds having a corresponding type 0
    signature" used for the Extended Data Fig. 2/5 comparisons in the CC
    Protocols paper.

    Parameters
    ----------
    cc : chemicalchecker.core.chemcheck.ChemicalChecker
        CC instance the dataset belongs to.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    set of str
        Compound keys present in the dataset's sign0.
    """
    sign0 = cc.get_signature("sign0", "full", dataset_code)
    return set(sign0.keys)


def plot_input_compound_overlay(
    projection_sign3,
    keys_a: set[str],
    keys_b: set[str],
    label_a: str,
    label_b: str,
    output_path: Path,
    max_points: int = 8000,
    random_state: int = 0,
) -> None:
    """t-SNE overlay of genuine input-compound membership between two datasets.

    Mirrors Extended Data Fig. 2/5 in Comajuncosa-Creus et al.: shows
    whether the compounds a dataset actually has raw (sign0) data for
    populate previously uncovered regions of bioactivity space, or just add
    density around already-covered compounds. Both compound sets are
    projected using a single sign3 space (``projection_sign3``, normally the
    larger/extended dataset's, since sign3 covers virtually the whole CC
    universe) so the embedding is self-consistent -- mixing vectors from two
    independently-trained sign3 spaces would not be, since each space's
    embedding has its own arbitrary orientation.

    Parameters
    ----------
    projection_sign3 : chemicalchecker.core.signature_data.DataSignature
        The sign3 signature to fetch projection vectors from.
    keys_a, keys_b : set of str
        Sign0 (real, non-inferred) compound keys for dataset A and B.
    label_a, label_b : str
        Display labels for the two datasets.
    output_path : Path
        Where to save the figure (PNG).
    max_points : int, default 8000
        Cap on total points projected (subsampled if exceeded), matching the
        ~10k cap CC's own diagnosis plots use.
    random_state : int, default 0
        Seed for subsampling and the t-SNE fit.

    Raises
    ------
    RuntimeError
        If fewer than 50 compounds are found in ``projection_sign3`` after
        subsampling.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    shared = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    all_keys = sorted(shared) + sorted(only_a) + sorted(only_b)
    membership = (
        ["shared"] * len(shared)
        + [f"{label_a} only"] * len(only_a)
        + [f"{label_b} only"] * len(only_b)
    )

    rng = np.random.default_rng(random_state)
    if len(all_keys) > max_points:
        idx = rng.choice(len(all_keys), size=max_points, replace=False)
        all_keys = [all_keys[i] for i in idx]
        membership = [membership[i] for i in idx]

    membership_by_key = dict(zip(all_keys, membership))
    found_keys, vectors = projection_sign3.get_vectors(all_keys)
    if vectors is None or len(found_keys) < 50:
        n_found = 0 if vectors is None else len(found_keys)
        raise RuntimeError(
            f"Only {n_found} of {len(all_keys)} requested compounds were found "
            "in the projection sign3 space; cannot project."
        )
    ordered_membership = [membership_by_key[k] for k in found_keys]

    embedding = TSNE(n_components=2, init="pca", random_state=random_state).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(6, 6))
    for group in sorted(set(ordered_membership)):
        mask = np.array([m == group for m in ordered_membership])
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.6, label=group)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Input-compound membership overlay")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_comparison(
    local_cc_dir_a: Path,
    dataset_a: str,
    local_cc_dir_b: Path,
    dataset_b: str,
    label_a: str,
    label_b: str,
    output_dir: Path,
    pval: float = 0.01,
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run the full space-vs-space comparison and write outputs to disk.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B (may be the
        same directory if both dataset codes live in one instance).
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.
    label_a, label_b : str
        Display labels used in tables/plot legends.
    output_dir : Path
        Directory to write comparison tables/figures to (created if missing).
    pval, n_random, n_subsamples, random_state
        Forwarded to :func:`shared_key_recapitulation`.
    make_plots : bool, default True
        Whether to save comparison figures in addition to tables.

    Returns
    -------
    dict
        Keys: ``summary_table``, ``across_roc_comparison`` (if available),
        ``shared_key_recapitulation``. Figures (when ``make_plots`` is True)
        are written to ``output_dir`` directly rather than returned:
        ``across_roc_scatter.png``, ``confidence_overlay.png``,
        ``headline_metrics_bar.png``, ``moa_atc_roc_overlay.png``,
        ``input_compound_tsne_overlay.png``,
        ``recap_roc_<label_a>_NN.png``, ``recap_roc_<label_b>_NN.png``.
        Each plot's failure is logged and skipped independently rather than
        aborting the whole run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # CC_CONFIG must already be set by the caller before importing chemicalchecker.
    from chemicalchecker import ChemicalChecker

    logger.info("Loading sign3 signatures for %s and %s ...", label_a, label_b)
    cc_a = ChemicalChecker(str(local_cc_dir_a), dbconnect=False)
    cc_b = cc_a if local_cc_dir_a == local_cc_dir_b else ChemicalChecker(str(local_cc_dir_b), dbconnect=False)
    sign3_a = cc_a.get_signature("sign3", "full", dataset_a)
    sign3_b = cc_b.get_signature("sign3", "full", dataset_b)

    logger.info("Building summary table from saved diagnosis artifacts ...")
    specs = [SpaceSpec(label_a, dataset_a), SpaceSpec(label_b, dataset_b)]
    local_cc_dirs = {label_a: local_cc_dir_a, label_b: local_cc_dir_b}
    summary_df = build_summary_table(specs, local_cc_dirs)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    results["summary_table"] = summary_df
    logger.info("Wrote summary table to %s", summary_path)

    logger.info("Comparing per-CC-space NN recapitulation (across_roc) ...")
    across_df = None
    try:
        across_df = compare_across_roc(local_cc_dir_a, dataset_a, local_cc_dir_b, dataset_b)
        across_path = output_dir / "across_roc_comparison.csv"
        across_df.to_csv(across_path, index=False)
        results["across_roc_comparison"] = across_df
        logger.info("Wrote across-CC-space comparison to %s", across_path)
    except (FileNotFoundError, TypeError) as exc:
        logger.warning("Skipping across_roc comparison: %s", exc)

    if make_plots and across_df is not None:
        plot_path = output_dir / "across_roc_scatter.png"
        try:
            plot_across_roc_scatter(across_df, plot_path, label_a, label_b)
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

    logger.info("Comparing confidence distributions ...")
    try:
        conf_df = compare_confidence(local_cc_dir_a, dataset_a, local_cc_dir_b, dataset_b)
        if make_plots:
            plot_path = output_dir / "confidence_overlay.png"
            try:
                plot_confidence_overlay(conf_df, plot_path)
                logger.info("Wrote %s", plot_path)
            except Exception:
                logger.exception("Failed to write %s", plot_path)
    except FileNotFoundError as exc:
        logger.warning("Skipping confidence comparison: %s", exc)

    if make_plots:
        plot_path = output_dir / "headline_metrics_bar.png"
        try:
            plot_headline_metrics_bar(summary_df, plot_path)
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

        plot_path = output_dir / "moa_atc_roc_overlay.png"
        try:
            plot_validation_roc_overlay(
                local_cc_dir_a, dataset_a, label_a, local_cc_dir_b, dataset_b, label_b, plot_path
            )
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

        plot_path = output_dir / "input_compound_tsne_overlay.png"
        try:
            sign0_keys_a = get_sign0_keys(cc_a, dataset_a)
            sign0_keys_b = get_sign0_keys(cc_b, dataset_b)
            projection_sign3, projection_label = (
                (sign3_a, label_a) if len(sign3_a.keys) >= len(sign3_b.keys) else (sign3_b, label_b)
            )
            logger.info("Projecting input-compound overlay using %s's sign3 space.", projection_label)
            plot_input_compound_overlay(
                projection_sign3, sign0_keys_a, sign0_keys_b, label_a, label_b, plot_path
            )
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

    logger.info(
        "Running shared-compound NN recapitulation (pval=%.4f, n_random=%d, n_subsamples=%d) ...",
        pval,
        n_random,
        n_subsamples,
    )
    shared_keys, vec_a, vec_b = get_shared_vectors(sign3_a, sign3_b)
    recap = shared_key_recapitulation(
        vec_a,
        vec_b,
        n_shared_compounds=len(shared_keys),
        pval=pval,
        n_random=n_random,
        n_subsamples=n_subsamples,
        random_state=random_state,
    )
    recap_path = output_dir / "shared_key_recapitulation.json"
    with open(recap_path, "w", encoding="utf-8") as fh:
        json.dump(recap, fh, indent=2)
    results["shared_key_recapitulation"] = recap
    logger.info("Wrote shared-key recapitulation results to %s", recap_path)
    logger.info(
        "%s NN recapitulated by %s: AUROC=%.3f +/- %.3f (n=%d shared compounds)",
        label_a,
        label_b,
        recap["auroc_a_recap_by_b"],
        recap["std_a_recap_by_b"],
        recap["n_shared_compounds"],
    )
    logger.info(
        "%s NN recapitulated by %s: AUROC=%.3f +/- %.3f",
        label_b,
        label_a,
        recap["auroc_b_recap_by_a"],
        recap["std_b_recap_by_a"],
    )

    if make_plots:
        plot_path = output_dir / f"recap_roc_{label_a}_NN.png"
        try:
            plot_recapitulation_roc(
                vec_a, vec_b, label_a, label_b, plot_path,
                n_random=n_random, n_subsamples=n_subsamples, random_state=random_state,
            )
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

        plot_path = output_dir / f"recap_roc_{label_b}_NN.png"
        try:
            plot_recapitulation_roc(
                vec_b, vec_a, label_b, label_a, plot_path,
                n_random=n_random, n_subsamples=n_subsamples, random_state=random_state,
            )
            logger.info("Wrote %s", plot_path)
        except Exception:
            logger.exception("Failed to write %s", plot_path)

    return results