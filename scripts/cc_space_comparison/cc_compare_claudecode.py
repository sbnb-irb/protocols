"""Core logic for comparing two Chemical Checker bioactivity spaces.

This is a reworked version of ``cc_compare.py``, which is kept unchanged
alongside it for reference. The differences that matter scientifically --
rather than stylistically -- are recorded under "Artifact shapes" below and
in ``RESULTS.md``; the short version is that the headline MoA/ATC AUROC was
being read from an artifact that reports a different metric and the opposite
winner, and that the nearest-neighbor cutoff was estimated per-subsample
instead of from a background distribution as the protocol specifies.

This module implements the space-vs-space validation analyses described in
Comajuncosa-Creus et al. (Nat. Protoc. 2025) -- the shared-compound
nearest-neighbor recapitulation test (Fig. 3a/b, Fig. 3g,h and Extended
Data Fig. 8g,h) and side-by-side comparisons of the diagnosis artifacts
that ``signature.diagnosis().canvas()`` already writes to disk.

No signature fitting happens here -- see ``cc_pipeline.py`` for that.

Artifact shapes: verified, not assumed
--------------------------------------
Every artifact this module reads was opened with ``pickle.load`` on a real
``local_CC_D6`` instance and its structure confirmed for *both* compared
datasets before the parsing code was written. The confirmed shapes are
recorded in each loader's docstring. The notes below record the findings
that changed what this module computes, because several of them are
conclusion-flipping rather than cosmetic.

**There are two different MoA/ATC AUROCs in a CC diagnosis run, and they
disagree about which space is better.** They are both real, they measure
different things, and this module reports both, separately labeled:

``moa_roc.pkl`` / ``atc_roc.pkl`` -- the *neighborhood* metric
    Written by ``Diagnosis.cross_roc()`` against B1.001 / E1.001. It takes
    10,000 shared molecules, calls each molecule's k=5 nearest neighbors
    *in B1/E1 sign3* a positive pair, samples an equal number of random
    negatives, then scores every pair by cosine distance *in the space
    being diagnosed*. This is the number printed in the diagnosis canvas
    title, and it is the Fig. 3 / Ext. Data Fig. 8 NN-recapitulation
    family. Exported as ``moa_auroc_neighborhood`` / ``atc_auroc_neighborhood``.

``validation_stats.json`` / ``*_auc_validation.tsv`` -- the *annotation* metric
    Written by ``signature_base.validate()`` -> ``Plot.vector_validation()``.
    It reads a **fixed external validation set** from CC's own
    ``tests/validation_sets/``, builds S = compound pairs sharing a MoA/ATC
    annotation and D = pairs that do not, and computes both a KS test
    (``*_ks_d`` / ``*_ks_p``) and an ROC separating S from D by cosine
    distance in this space. Exported as ``moa_auroc_annotation`` /
    ``atc_auroc_annotation``.

Earlier versions of this module sourced the headline MoA/ATC number from
``validation_stats.json``, which reports the *opposite* winner from the
diagnosis canvas. The neighborhood metric is the paper-faithful one and is
treated as the headline here; the annotation metric is kept alongside it
because its dissent is a real result worth surfacing, not a bug to hide.
One honest caveat in the other direction: the neighborhood metric is
computed on an *unseeded* random subsample, while the annotation metric is
deterministic on a fixed external pair set, so the annotation metric is the
better-controlled of the two even though it is not the paper's headline.

**Metrics that are NOT discriminative between two sign3 spaces**, confirmed
from the CC source rather than inferred, and therefore reported with an
explicit caveat instead of as evidence:

``pct_outliers``
    ``Diagnosis.outliers()`` fits ``IsolationForest(contamination=0.1)`` --
    a hardcoded rate. Exactly 1,000 of 10,000 keys are flagged in every
    dataset, so the percentage carries no information at all. Only the
    anomaly *score* distribution differs. Note also that the stored
    ``scores`` follow the scikit-learn convention (negative = outlier)
    while CC's own plotter negates them (``scs = -results["scores"]``), so
    Table S2's "scores above 0 are considered outliers" describes the
    *plotted* sign. Getting this backwards reports the inlier percentage.

``moa_cov`` / ``atc_cov``
    The ``frac`` return of ``vector_validation`` -- coverage of the fixed
    external validation file, not of each dataset's own compounds. Both
    sign3 spaces span essentially the whole CC universe, so these come out
    bit-for-bit identical. ``across_coverage.pkl`` says the same thing:
    ``vs_overlap == 1.0`` for all 25 CC spaces in both datasets, which per
    Table S2 row "CC wrt Sign" is the *expected pass* for a type III
    signature, not a point of difference.

**Two artifacts need defensive handling.** ``*_auc_validation.tsv`` is
written by a bare ``f.write("%f\\t%f\\n")`` loop and has **no header row**,
so it must be read with ``header=None``. And ``global_ranks_agreement.pkl``
can contain NaN for a compound present in no other CC space (RBO is
undefined there), which silently turns a plain ``np.mean`` into NaN -- all
reductions over it use the ``nan``-safe variants.

Comparability caveat
--------------------
Within one dataset, the diagnosis artifacts all describe an identical
10,000-key subsample. *Between* datasets they do not: the two runs drew
different subsamples (only ~21% overlap for the D6.002/D6.007 pair). Every
10k-based comparison here is therefore unpaired -- two independent draws
from the same ~1.2M-compound universe. That is fine for comparing
distributions, but small deltas should not be over-read.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import paired_cosine_distances

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presentation palette
# ---------------------------------------------------------------------------
# One muted, colorblind-safe pair identifies space A vs space B in *every*
# figure this module writes, so a reader moving between slides never has to
# re-learn which color is which. Validated (OKLab dCVD / chroma / contrast
# against a light surface): worst adjacent CVD delta-E 24.7, normal-vision
# 33.6, both well above the 8 / 15 floors.
SPACE_COLORS: tuple[str, str] = ("#2a78d6", "#eb6834")

# Two steps of a single blue ramp for the p-value cutoff curves *within* one
# recapitulation plot. Deliberately not the A/B pair: those two curves are
# two cutoffs on one comparison, not two spaces, and reusing the space colors
# there would imply a comparison that isn't being drawn.
CUTOFF_COLORS: tuple[str, str] = ("#2ca02c", "#5a8ad6")

# Neutral ink for context marks (the y=x diagonal, the chance line, and the
# "present in both" group in the membership overlay). Context is never a
# series, so it never takes a series color.
NEUTRAL_INK: str = "#8a8a80"


# The six signature-type combinations of Fig. 3b: nearest neighbours are
# defined at the lower (less abstract) signature and the higher one is scored
# on recovering them. Ordered lower-to-higher so the figure reads as a
# progression through the abstraction ladder.
SIGNATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("sign0", "sign1"),
    ("sign0", "sign2"),
    ("sign0", "sign3"),
    ("sign1", "sign2"),
    ("sign1", "sign3"),
    ("sign2", "sign3"),
)


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


# ---------------------------------------------------------------------------
# Locating artifacts on disk
# ---------------------------------------------------------------------------


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
        Signature type, e.g. ``"sign0"``, ``"sign2"``, ``"sign3"``.

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


def load_roc_artifact(
    local_cc_dir: Path, dataset_code: str, kind: str, sign_type: str = "sign3"
) -> dict[str, Any]:
    """Load ``moa_roc.pkl`` / ``atc_roc.pkl`` -- the canvas MoA/ATC ROC curve.

    This is the artifact that actually backs the AUROC printed in the
    diagnosis canvas title, and therefore the paper-faithful source for the
    headline MoA/ATC number. See this module's docstring for why it differs
    from ``validation_stats.json``.

    Confirmed shape (both D6.002 and D6.007)::

        {'fpr': ndarray, 'tpr': ndarray, 'auc': float,
         'precision': ndarray, 'recall': ndarray,
         'average_precision_score': float}

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    kind : str
        Either ``"moa"`` (recapitulation of B1.001) or ``"atc"`` (E1.001).
    sign_type : str, default "sign3"
        Signature type the diagnosis was run on.

    Returns
    -------
    dict
        The unpickled ROC/PR result dictionary.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"moa"`` or ``"atc"``.
    TypeError
        If the artifact is not a dict carrying the expected ``fpr``/``tpr``/
        ``auc`` fields -- i.e. if this CC version's shape differs from the
        one this loader was verified against.
    """
    if kind not in {"moa", "atc"}:
        raise ValueError(f"kind must be 'moa' or 'atc', got {kind!r}")
    raw = load_diag_artifact(local_cc_dir, dataset_code, f"{kind}_roc", sign_type)
    required = {"fpr", "tpr", "auc"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise TypeError(
            f"Unexpected {kind}_roc.pkl shape for {dataset_code}: expected a dict "
            f"containing {sorted(required)}, found {found}. Inspect it with "
            "pickle.load and update load_roc_artifact()."
        )
    return raw


def load_pair_validation_curve(
    local_cc_dir: Path, dataset_code: str, kind: str, sign_type: str = "sign3"
) -> pd.DataFrame:
    """Load the *annotation-metric* ROC curve from ``*_auc_validation.tsv``.

    This is the curve behind ``validation_stats.json``'s ``moa_auc`` /
    ``atc_auc`` -- the same-annotation vs different-annotation pair
    separation test, NOT the canvas neighborhood metric. Use
    :func:`load_roc_artifact` for the latter.

    The file is written by a bare ``f.write("%f\\t%f\\n")`` loop in
    ``Plot.roc_curve_plot`` and therefore has **no header row**. Reading it
    with pandas' default ``header='infer'`` silently promotes the first data
    point to a column name, which is what made every earlier attempt to plot
    this curve fail.

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
        Two columns, ``fpr`` and ``tpr``, in the order written.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"moa"`` or ``"atc"``.
    FileNotFoundError
        If the validation table does not exist.
    """
    if kind not in {"moa", "atc"}:
        raise ValueError(f"kind must be 'moa' or 'atc', got {kind!r}")
    stats_dir = _dataset_signature_dir(local_cc_dir, dataset_code, sign_type) / "stats"
    path = stats_dir / f"{kind}_{sign_type}_auc_validation.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Validation table not found: {path}")
    return pd.read_csv(path, sep="\t", header=None, names=["fpr", "tpr"])


def load_validation_stats(
    local_cc_dir: Path, dataset_code: str, sign_type: str = "sign3"
) -> dict[str, Any]:
    """Load ``stats/validation_stats.json`` for a dataset.

    Carries the *annotation* metric (see module docstring): ``molecules``,
    ``{moa,atc}_ks_d``, ``{moa,atc}_ks_p``, ``{moa,atc}_auc`` and
    ``{moa,atc}_cov``.

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

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    stats_dir = _dataset_signature_dir(local_cc_dir, dataset_code, sign_type) / "stats"
    path = stats_dir / "validation_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"validation_stats.json not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Artifact -> scalar/array coercion
# ---------------------------------------------------------------------------


def _coerce_1d_array(raw: Any, name: str) -> np.ndarray:
    """Best-effort coercion of a diagnosis artifact into a flat 1D array.

    Confirmed dict payloads for this CC build::

        confidences.pkl -> {'keys', 'confidences', 'x', 'y'}  (uses 'confidences')
        outliers.pkl    -> {'scores', 'pred'}                 (uses 'scores')

    ``'x'``/``'y'`` in ``confidences.pkl`` are the precomputed density-curve
    coordinates for the canvas panel, not per-compound values -- the
    per-compound array is ``'confidences'``, which the ``name``-based lookup
    below resolves. For ``outliers.pkl`` the continuous ``'scores'`` is taken
    rather than the thresholded ``'pred'``; see :func:`outlier_summary` for
    why the thresholded call is not usable as a comparison metric.

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
            "(pickle.load) and extend _coerce_1d_array()."
        )
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw).ravel()
    raise TypeError(f"Unrecognized '{name}' payload type: {type(raw)!r}.")


def _coerce_across_roc(raw: Any) -> pd.DataFrame:
    """Coerce an ``across_roc.pkl`` payload into a tidy ``cc_space``/``auroc`` frame.

    Confirmed shape: a ``dict`` keyed by CC dataset code (25 entries, the
    exemplary space of each coordinate), each value itself a ``dict`` with a
    full ROC/PR curve plus scalars -- ``{'fpr', 'tpr', 'auc', 'precision',
    'recall', 'average_precision_score'}``. Only the scalar ``'auc'`` is
    extracted; the curves are deliberately left behind, because writing a
    whole array into a DataFrame cell produces an unusable truncated repr
    once the frame is exported to CSV.

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
        If ``raw`` matches none of the recognized shapes.
    """
    if isinstance(raw, pd.DataFrame):
        df = raw.reset_index()
        df = df.rename(columns={df.columns[0]: "cc_space", df.columns[1]: "auroc"})
        return df[["cc_space", "auroc"]]
    if isinstance(raw, dict):
        values = list(raw.values())
        if values and isinstance(values[0], dict) and "auc" in values[0]:
            return pd.DataFrame(
                {"cc_space": list(raw.keys()), "auroc": [v["auc"] for v in values]}
            )
        if values and all(isinstance(v, (int, float, np.floating, np.integer)) for v in values):
            return pd.DataFrame({"cc_space": list(raw.keys()), "auroc": values})
        raise TypeError(
            "Unrecognized across_roc.pkl dict shape (first value type: "
            f"{type(values[0]) if values else None!r}). Inspect it manually "
            "(pickle.load) and extend _coerce_across_roc()."
        )
    raise TypeError(
        f"Unrecognized across_roc.pkl payload type: {type(raw)!r}. "
        "Inspect it manually (pickle.load) and extend _coerce_across_roc()."
    )


# ---------------------------------------------------------------------------
# Per-space metric extraction
# ---------------------------------------------------------------------------


def space_dimensions(local_cc_dir: Path, dataset_code: str) -> dict[str, int]:
    """Read this space's own sign3 key/feature counts from ``dimensions.pkl``.

    ``dimensions.pkl`` maps each of the 25 CC coordinates to its
    ``{'keys', 'features'}`` plus a ``'MY'`` entry describing *this* space.
    The ``'MY'`` entry is the only part that differs between two datasets in
    the same CC instance; the other 25 are reference dimensions.

    This exists because the number of rows in a diagnosis artifact is the
    subsample size (10,000), not the space's key count -- reporting the
    former as ``n_keys`` understates a ~1.2M-compound space by two orders of
    magnitude.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``{"n_keys": int, "n_features": int}``.

    Raises
    ------
    KeyError
        If ``dimensions.pkl`` has no ``'MY'`` entry.
    """
    dims = load_diag_artifact(local_cc_dir, dataset_code, "dimensions")
    if "MY" not in dims:
        raise KeyError(
            f"dimensions.pkl for {dataset_code} has no 'MY' entry "
            f"(keys: {sorted(dims)[:8]}...). Cannot read this space's own size."
        )
    return {"n_keys": int(dims["MY"]["keys"]), "n_features": int(dims["MY"]["features"])}


def outlier_summary(local_cc_dir: Path, dataset_code: str) -> dict[str, float]:
    """Summarize ``outliers.pkl``, respecting CC's sign convention.

    Two things about this artifact make the naive summary misleading, both
    confirmed from the CC source rather than guessed:

    1. ``Diagnosis.outliers()`` fits ``IsolationForest(contamination=0.1)``.
       The flagged fraction is therefore pinned at exactly 10% for *every*
       dataset and cannot distinguish two spaces. It is reported here only
       so the constancy is visible, never as evidence.
    2. The stored ``scores`` use scikit-learn's ``decision_function``
       convention (negative = outlier), but CC's own plotter negates them
       (``scs = -results["scores"]``), so the canvas -- and Table S2's
       "scores above 0 are considered outliers" -- shows the flipped sign.
       This function returns the *plotted* (Table S2) orientation, where
       higher means more anomalous.

    The only informative quantity here is the anomaly-score distribution.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``pct_flagged_outliers`` (expected to be exactly 10.0),
        ``mean_anomaly_score`` and ``std_anomaly_score`` in the plotted
        orientation (higher = more anomalous).
    """
    raw = load_diag_artifact(local_cc_dir, dataset_code, "outliers")
    scores = _coerce_1d_array(raw, "outliers")
    # Flip to the plotted/Table S2 orientation: higher = more anomalous.
    anomaly = -np.asarray(scores, dtype=float)
    pct_flagged = float(np.mean(anomaly > 0) * 100)
    if abs(pct_flagged - 10.0) > 0.5:
        # Worth knowing about: it would mean this CC build changed the
        # hardcoded contamination, which changes how to read the metric.
        logger.warning(
            "%s: outlier fraction is %.2f%%, not the expected 10%% from "
            "IsolationForest(contamination=0.1).",
            dataset_code,
            pct_flagged,
        )
    return {
        "pct_flagged_outliers": pct_flagged,
        "mean_anomaly_score": float(np.mean(anomaly)),
        "std_anomaly_score": float(np.std(anomaly)),
    }


def redundancy_summary(local_cc_dir: Path, dataset_code: str) -> dict[str, float]:
    """Percentage of signatures that are exact duplicates of another.

    Per Table S2 ("Redund"), lower redundancy is the favorable outcome: it
    means less overlap among compound signatures. ``redundancy.pkl`` stores
    ``{'n_full', 'n_ref', 'counts'}``, where ``n_ref`` is the size of the
    non-redundant set; the percentage is ``1 - n_ref / n_full``.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``n_full``, ``n_non_redundant`` and ``pct_redundant``.
    """
    raw = load_diag_artifact(local_cc_dir, dataset_code, "redundancy")
    n_full, n_ref = int(raw["n_full"]), int(raw["n_ref"])
    return {
        "n_full": n_full,
        "n_non_redundant": n_ref,
        "pct_redundant": float((1.0 - n_ref / n_full) * 100),
    }


def cluster_summary(local_cc_dir: Path, dataset_code: str) -> dict[str, float]:
    """DBSCAN cluster count / granularity / unclustered fraction.

    Included because Table S2 ("Cluster sizes") reads this in a direction
    that runs *opposite* to most other panels: "a large number of clusters
    with low proportions of compounds each may indicate a continuous
    representation of the chemical space, capturing fine details in terms of
    compound diversity", while "few and large clusters typically indicate a
    certain degree of redundancy". More clusters is therefore the favorable
    outcome here. Table S2 also notes that compounds assigned to no cluster
    (DBSCAN label ``-1``) "correspond to outliers", so the noise fraction is
    a genuine per-dataset outlier measure -- unlike ``pct_flagged_outliers``,
    which is pinned by a fixed contamination rate.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``n_clusters``, ``dbscan_epsilon``, ``pct_unclustered`` and
        ``largest_cluster_pct``.
    """
    raw = load_diag_artifact(local_cc_dir, dataset_code, "clusters")
    labels = np.asarray(raw["labels"])
    clustered = labels[labels >= 0]
    largest = 0.0
    if clustered.size:
        largest = float(np.bincount(clustered).max() / labels.size * 100)
    return {
        "n_clusters": int(raw["n_clusters"]),
        "dbscan_epsilon": float(raw["epsilon"]),
        "pct_unclustered": float(np.mean(labels == -1) * 100),
        "largest_cluster_pct": largest,
    }


def key_coverage_summary(local_cc_dir: Path, dataset_code: str) -> dict[str, float]:
    """How many other CC spaces this space's compounds also appear in.

    ``key_coverage.pkl['key_counts']`` gives, for each of the 10,000
    subsampled compounds, the number of the 25 CC spaces it is present in.
    This is the metric Table S2 ("Key Coverage (I)") ties directly to type
    III signature confidence: the good outcome is a high proportion of
    molecules present in at least 5-6 datasets, while "a high proportion of
    molecules present in less than 5 CC spaces typically indicat[es] little
    overlap with the CC universe (leading to unconfident type III
    signatures)".

    It is the most direct available answer to "did the newly added compounds
    connect to the rest of the CC universe, or land as isolated signatures".

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``mean_cc_spaces_per_compound``, ``median_cc_spaces_per_compound``,
        ``min_cc_spaces``, ``pct_in_fewer_than_5_spaces`` and
        ``n_in_zero_other_spaces``.
    """
    raw = load_diag_artifact(local_cc_dir, dataset_code, "key_coverage")
    counts = np.asarray(raw["key_counts"], dtype=float)
    return {
        "mean_cc_spaces_per_compound": float(np.mean(counts)),
        "median_cc_spaces_per_compound": float(np.median(counts)),
        "min_cc_spaces": float(np.min(counts)),
        "pct_in_fewer_than_5_spaces": float(np.mean(counts < 5) * 100),
        "n_in_zero_other_spaces": int(np.sum(counts == 0)),
    }


def ranks_agreement_summary(local_cc_dir: Path, dataset_code: str) -> dict[str, float]:
    """Rank-biased-overlap agreement of this space's neighbors with other CC spaces.

    Per Table S2 ("CC ranks agreement"), **low** agreement is the favorable
    outcome: "low RBO values typically indicate that the descriptors are
    correctly encoding the specificity of the current space, showing a
    certain degree of orthogonality to other CC spaces", and the good
    outcome is the distribution peaking near 0. A *higher* value here is
    therefore evidence of redundancy with the rest of the CC, not of
    quality.

    All reductions are ``nan``-safe on purpose. A compound present in no
    other CC space has no ranking to overlap with, so its RBO is NaN --
    D6.007 has exactly one such compound, and a plain ``np.mean`` over the
    array silently returns NaN for the whole space.

    Parameters
    ----------
    local_cc_dir : Path
        Root directory of the local Chemical Checker instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.

    Returns
    -------
    dict
        ``mean_ranks_agreement``, ``median_ranks_agreement``,
        ``max_ranks_agreement`` and ``n_undefined_ranks_agreement``.
    """
    raw = load_diag_artifact(local_cc_dir, dataset_code, "global_ranks_agreement")
    per_compound_mean = np.asarray(raw["mean"], dtype=float)
    n_nan = int(np.isnan(per_compound_mean).sum())
    if n_nan:
        logger.info(
            "%s: %d compound(s) have undefined CC ranks agreement (present in no "
            "other CC space); using nan-safe reductions.",
            dataset_code,
            n_nan,
        )
    return {
        "mean_ranks_agreement": float(np.nanmean(per_compound_mean)),
        "median_ranks_agreement": float(np.nanmedian(per_compound_mean)),
        "max_ranks_agreement": float(np.nanmean(np.asarray(raw["max"], dtype=float))),
        "n_undefined_ranks_agreement": n_nan,
    }


# ---------------------------------------------------------------------------
# Cross-space comparison tables
# ---------------------------------------------------------------------------


def build_summary_table(specs: list[SpaceSpec], local_cc_dirs: dict[str, Path]) -> pd.DataFrame:
    """One row per space with headline validation numbers pulled from disk.

    Collects the numbers shown in the diagnosis canvas titles
    programmatically instead of reading them off the figure, and keeps the
    two disagreeing MoA/ATC metrics side by side under explicit names:

    ``moa_auroc_neighborhood`` / ``atc_auroc_neighborhood``
        From ``moa_roc.pkl`` / ``atc_roc.pkl``. The canvas number, and the
        paper-faithful NN-recapitulation metric. **Treat as the headline.**
    ``moa_auroc_annotation`` / ``atc_auroc_annotation``
        From ``validation_stats.json``. Same-annotation vs different-
        annotation pair separation against a fixed external validation set.

    Coverage fields (``*_cov``) are deliberately excluded: they measure
    coverage of that fixed external file, come out bit-for-bit identical for
    any two whole-universe sign3 spaces, and so would only pad the table
    with a non-difference. See the module docstring.

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
        dims = space_dimensions(cc_dir, spec.dataset_code)

        row: dict[str, Any] = {
            "space": spec.label,
            "dataset_code": spec.dataset_code,
            # The space's real size, from dimensions.pkl['MY'] -- not the
            # 10,000-row diagnosis subsample the other artifacts describe.
            "n_keys": dims["n_keys"],
            "n_features": dims["n_features"],
            "n_diagnosis_subsample": int(len(confidences)),
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "pct_confidence_above_0.6": float(np.mean(confidences > 0.6) * 100),
        }
        for kind in ("moa", "atc"):
            row[f"{kind}_auroc_neighborhood"] = float(
                load_roc_artifact(cc_dir, spec.dataset_code, kind)["auc"]
            )
            row[f"{kind}_auroc_annotation"] = float(stats[f"{kind}_auc"])
            row[f"{kind}_ks_d"] = float(stats[f"{kind}_ks_d"])
        row.update(outlier_summary(cc_dir, spec.dataset_code))
        rows.append(row)
    return pd.DataFrame(rows)


def build_diagnosis_extras_table(
    specs: list[SpaceSpec], local_cc_dirs: dict[str, Path]
) -> pd.DataFrame:
    """Per-space table of the diagnosis panels not covered by the summary table.

    Covers redundancy, DBSCAN clustering, CC key coverage and CC ranks
    agreement -- the ``Table S2`` panels that carry a documented good/bad
    interpretation and that actually differ between two sign3 spaces. Each
    contributing function's docstring records the direction Table S2 reads
    the metric in, which is not uniformly "higher is better": more clusters
    and *lower* ranks agreement are both favorable outcomes.

    Parameters
    ----------
    specs : list of SpaceSpec
        Spaces to summarize.
    local_cc_dirs : dict
        Maps each ``SpaceSpec.label`` to its local CC instance directory.

    Returns
    -------
    pandas.DataFrame
        One row per space. Any panel that fails to load is logged and left
        out rather than aborting the whole table.
    """
    extractors = {
        "redundancy": redundancy_summary,
        "clusters": cluster_summary,
        "key_coverage": key_coverage_summary,
        "ranks_agreement": ranks_agreement_summary,
    }
    rows = []
    for spec in specs:
        cc_dir = local_cc_dirs[spec.label]
        row: dict[str, Any] = {"space": spec.label, "dataset_code": spec.dataset_code}
        for name, fn in extractors.items():
            try:
                row.update(fn(cc_dir, spec.dataset_code))
            except (FileNotFoundError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping '%s' panel for %s: %s", name, spec.dataset_code, exc
                )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_across_roc(
    local_cc_dir_a: Path, dataset_a: str, local_cc_dir_b: Path, dataset_b: str
) -> pd.DataFrame:
    """Compare per-CC-space nearest-neighbor recapitulation AUROCs.

    Reproduces the "ROC across CC" diagnosis panel for both spaces and joins
    them on the CC dataset code, so each CC space becomes one row with an
    AUROC from each compared space.

    The join is an **outer** join on purpose. An inner join would silently
    drop any CC space that only one of the two diagnosis runs covered, which
    is exactly the kind of difference the reader needs told about rather
    than hidden -- so unmatched codes are kept (with NaN on the missing
    side) and logged.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B.
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.

    Returns
    -------
    pandas.DataFrame
        Columns: ``cc_space``, ``auroc_a``, ``auroc_b``, ``delta_b_minus_a``,
        sorted by CC space code.
    """
    df_a = _coerce_across_roc(load_diag_artifact(local_cc_dir_a, dataset_a, "across_roc"))
    df_b = _coerce_across_roc(load_diag_artifact(local_cc_dir_b, dataset_b, "across_roc"))
    merged = df_a.merge(df_b, on="cc_space", how="outer", suffixes=("_a", "_b"))
    merged = merged.sort_values("cc_space").reset_index(drop=True)

    unmatched = merged.loc[merged[["auroc_a", "auroc_b"]].isna().any(axis=1), "cc_space"]
    if len(unmatched):
        logger.warning(
            "%d CC space(s) present in only one diagnosis run, kept with NaN: %s",
            len(unmatched),
            ", ".join(unmatched.tolist()),
        )
    merged["delta_b_minus_a"] = merged["auroc_b"] - merged["auroc_a"]
    return merged


def compare_confidence(
    local_cc_dir_a: Path,
    dataset_a: str,
    label_a: str,
    local_cc_dir_b: Path,
    dataset_b: str,
    label_b: str,
) -> pd.DataFrame:
    """Stack confidence-score distributions from two spaces into one frame.

    Rows are tagged with the *display label* rather than the dataset code so
    that the confidence figure's legend matches every other figure in the
    output directory.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B.
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.
    label_a, label_b : str
        Display labels for the two compared spaces.

    Returns
    -------
    pandas.DataFrame
        Columns: ``space``, ``confidence``.
    """
    conf_a = _coerce_1d_array(
        load_diag_artifact(local_cc_dir_a, dataset_a, "confidences"), "confidences"
    )
    conf_b = _coerce_1d_array(
        load_diag_artifact(local_cc_dir_b, dataset_b, "confidences"), "confidences"
    )
    return pd.DataFrame(
        {
            "space": [label_a] * len(conf_a) + [label_b] * len(conf_b),
            "confidence": np.concatenate([conf_a, conf_b]),
        }
    )


# ---------------------------------------------------------------------------
# Nearest-neighbor recapitulation (paper methodology)
# ---------------------------------------------------------------------------


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


def get_sign0_keys(cc, dataset_code: str) -> set[str]:
    """Return the set of compound keys with real (non-inferred) sign0 data.

    This is the only honest definition of "compounds this dataset actually
    measured". sign3 is inferred for essentially the entire CC universe
    regardless of whether a compound had real data in this space, so a
    membership overlay built from sign3 keys would mark ~1.2M compounds as
    belonging to both datasets and show nothing. It matches the "compounds
    having a corresponding type 0 signature" notion used for the Extended
    Data Fig. 2/5 comparisons in the CC Protocols paper.

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


def sign0_membership_summary(
    cc_a, dataset_a: str, label_a: str, cc_b, dataset_b: str, label_b: str
) -> dict[str, Any]:
    """Compare the two datasets' real (sign0) input compounds and features.

    Reports the compound-set relationship *and* the raw feature counts,
    because for a proteomics space the second is as much a part of "did the
    extension help" as the first: integrating more datasets by keeping only
    proteins shared across all of them buys compounds at the cost of
    measured features, and a comparison that reports only the compound gain
    would describe half the change.

    Parameters
    ----------
    cc_a, cc_b : chemicalchecker.core.chemcheck.ChemicalChecker
        CC instances the two datasets belong to.
    dataset_a, dataset_b : str
        CC dataset codes.
    label_a, label_b : str
        Display labels for the two spaces.

    Returns
    -------
    dict
        Compound counts, overlap/exclusive counts, whether one set is a
        strict subset of the other, and each space's sign0 feature count.
    """
    sign0_a = cc_a.get_signature("sign0", "full", dataset_a)
    sign0_b = cc_b.get_signature("sign0", "full", dataset_b)
    keys_a, keys_b = set(sign0_a.keys), set(sign0_b.keys)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_input_compounds_a": len(keys_a),
        "n_input_compounds_b": len(keys_b),
        "n_features_a": int(sign0_a.shape[1]),
        "n_features_b": int(sign0_b.shape[1]),
        "n_shared_input_compounds": len(keys_a & keys_b),
        "n_only_a": len(keys_a - keys_b),
        "n_only_b": len(keys_b - keys_a),
        "a_is_subset_of_b": keys_a.issubset(keys_b),
        "b_is_subset_of_a": keys_b.issubset(keys_a),
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _pyplot():
    """Import pyplot with a headless backend and this module's figure style.

    Importing ``chemicalchecker`` pulls in a seaborn style that switches axis
    grids on globally, so matplotlib's own default cannot be relied on here.
    These figures are headed for slides, where a grid behind the marks is
    noise competing with the data, so it is turned off explicitly.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module, configured.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _clean_axes(ax) -> None:
    """Strip an axes down to data, ticks and labels.

    Drops the grid (which the seaborn style imported by ``chemicalchecker``
    turns on) and the top/right spines, so the frame recedes and the marks
    carry the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle.
    """
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def _savefig_atomic(fig, output_path: Path, dpi: int = 300) -> None:
    """Save a matplotlib figure atomically (write to a temp file, then rename).

    A plain ``fig.savefig(path)`` can leave a truncated, misleadingly
    *present* zero-byte file at ``path`` if the process is interrupted
    mid-write. Writing to a sibling ``.tmp`` file and only then calling
    ``os.replace`` (atomic on POSIX) means an interruption leaves either a
    complete file or no file at all, never a broken one where a good one
    should be.

    This is a safety net, not the fix: the underlying cause was almost
    certainly peak memory, now addressed at source by the ``max_pool``
    subsampling in :func:`get_shared_vectors`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    output_path : Path
        Final destination path.
    dpi : int, default 300
        Resolution to save at.
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    # matplotlib infers the format from the extension, which ".tmp" breaks.
    fig.savefig(tmp_path, dpi=dpi, format=output_path.suffix.lstrip("."))
    os.replace(tmp_path, output_path)


def _space_colors(n: int = 2) -> list[str]:
    """Return the fixed space-identity colors, in A-then-B order.

    Colors are assigned by position (space A always the first hue, space B
    always the second) and never cycled or sorted, so the same space keeps
    the same color in every figure regardless of its label text.

    Parameters
    ----------
    n : int, default 2
        Number of colors needed.

    Returns
    -------
    list of str
        Hex color codes.
    """
    palette = list(SPACE_COLORS)
    while len(palette) < n:
        palette.extend(SPACE_COLORS)
    return palette[:n]


def _space_legend(fig, spaces: Sequence[str], colors: Sequence[str]) -> None:
    """Identify the spaces once, with a figure-level legend.

    In a grid of small multiples every panel compares the same two spaces, so
    tick-labelling each one repeats the same two strings a dozen times and
    forces them to rotate to fit. One legend for the whole figure says it
    once, and hands the panels their width back.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to attach the legend to.
    spaces : sequence of str
        Space labels, in the same order the bar colors were assigned.
    colors : sequence of str
        Their colors, positionally matched to ``spaces``.
    """
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=c, label=sp) for sp, c in zip(spaces, colors)],
        loc="upper center",
        ncol=len(spaces),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0),
    )


def _annotate_bars(ax, bars, values: Sequence[float], fmt: str = "{:.3f}") -> None:
    """Write each bar's exact value directly above it.

    Several of these comparisons come down to differences of 0.01-0.02,
    which is below what anyone can read off bar heights. The number on the
    bar is the data; the bar is the orientation. Labels use ordinary text
    ink rather than the series color, so identity stays with the mark.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the bars were drawn on.
    bars : matplotlib.container.BarContainer
        The bars returned by ``ax.bar``.
    values : sequence of float
        Values the bars encode.
    fmt : str, default "{:.3f}"
        Format string applied to each value.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return
    span = (max(finite) - min(finite)) or abs(max(finite)) or 1.0
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        # A negative bar grows downwards, so "above the end of the bar" is
        # below it. Anchoring on the wrong side drops the label inside the
        # bar, where it sits on the series color at poor contrast.
        below = value < 0
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, -4 if below else 3),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            fontsize=8,
            color="#333330",
        )
    # Room for the labels at whichever end(s) of the axis they sit.
    bottom, top = ax.get_ylim()
    ax.set_ylim(
        bottom - (span * 0.18 if min(finite) < 0 else 0.0),
        top + (span * 0.18 if max(finite) > 0 else 0.0),
    )


def plot_headline_metrics_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Small-multiple bar charts of the headline validation metrics.

    Each metric gets its own subplot and its own y-axis. A single shared
    axis fails badly here because the summary table mixes 0-1 AUROC/
    confidence scales with ~1e6 compound counts, which flattens every
    meaningful bar to invisibility -- count-like columns are therefore
    excluded outright rather than rescaled, since they are sizes, not
    validation metrics.

    KS-test p-values are excluded for a different reason: at ~1.2M
    compounds they underflow to exactly 0.0 regardless of effect size, so a
    bar height would encode nothing while looking like it encoded something.
    The KS D-statistic is the effect size and is kept.

    The two MoA/ATC metrics are shown side by side under their distinct
    names so their disagreement is visible rather than resolved by picking
    one -- see the module docstring.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of :func:`build_summary_table`.
    output_path : Path
        Where to save the figure (PNG).

    Raises
    ------
    ValueError
        If no plottable metric columns are found.
    """
    plt = _pyplot()

    # Explicit allowlist, in reading order: the two headline metrics first,
    # their dissenting counterparts next, then the supporting distributions.
    preferred = [
        "moa_auroc_neighborhood",
        "atc_auroc_neighborhood",
        "moa_auroc_annotation",
        "atc_auroc_annotation",
        "mean_confidence",
        "median_confidence",
        "moa_ks_d",
        "atc_ks_d",
        "mean_anomaly_score",
        "pct_flagged_outliers",
    ]
    metric_cols = [
        c for c in preferred
        if c in summary_df.columns and pd.api.types.is_numeric_dtype(summary_df[c])
    ]
    if not metric_cols:
        raise ValueError(
            "No plottable metric columns found in summary_df "
            f"(columns available: {list(summary_df.columns)})."
        )

    # Parentheticals are kept only where they identify *which* metric a panel
    # shows (the two MoA/ATC variants would otherwise be indistinguishable).
    titles = {
        "moa_auroc_neighborhood": "MoA AUROC\n(neighborhood, canvas)",
        "atc_auroc_neighborhood": "ATC AUROC\n(neighborhood, canvas)",
        "moa_auroc_annotation": "MoA AUROC\n(annotation set)",
        "atc_auroc_annotation": "ATC AUROC\n(annotation set)",
        "mean_confidence": "Mean confidence",
        "median_confidence": "Median confidence",
        "moa_ks_d": "MoA KS D",
        "atc_ks_d": "ATC KS D",
        "mean_anomaly_score": "Mean anomaly score",
        "pct_flagged_outliers": "Outliers flagged (%)",
    }

    ncols = min(5, len(metric_cols))
    nrows = int(np.ceil(len(metric_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    spaces = summary_df["space"].tolist()
    colors = _space_colors(len(spaces))
    for ax, col in zip(axes_flat, metric_cols):
        values = summary_df[col].to_numpy(dtype=float)
        bars = ax.bar(range(len(spaces)), values, color=colors, width=0.62)
        # Identity comes from the figure legend, not from repeating the same
        # two labels under every panel.
        ax.set_xticks([])
        ax.set_title(titles.get(col, col), fontsize=9)
        fmt = "{:.1f}" if col == "pct_flagged_outliers" else "{:.4f}"
        _annotate_bars(ax, bars, values, fmt=fmt)
        _clean_axes(ax)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes_flat[len(metric_cols):]:
        ax.axis("off")

    _space_legend(fig, spaces, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_moa_atc_roc_overlay(
    local_cc_dir_a: Path,
    dataset_a: str,
    label_a: str,
    local_cc_dir_b: Path,
    dataset_b: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Overlay both spaces' MoA and ATC ROC curves, for both metrics.

    Four panels: MoA and ATC, each shown under the *neighborhood* metric
    (from ``moa_roc.pkl`` / ``atc_roc.pkl``, the diagnosis canvas number)
    and the *annotation* metric (from ``*_auc_validation.tsv``). Putting
    them on one figure is the point -- the two metrics disagree about which
    space wins, and that disagreement is a result, not something to resolve
    by choosing a favorite.

    This replaces an earlier version that read the TSV with inferred column
    names and failed outright; had it worked, it would have plotted only the
    annotation metric while being labeled as the canvas one.

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
    plt = _pyplot()

    spaces = ((local_cc_dir_a, dataset_a, label_a), (local_cc_dir_b, dataset_b, label_b))
    colors = _space_colors(2)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9))
    for row, metric in enumerate(("neighborhood", "annotation")):
        for col, kind in enumerate(("moa", "atc")):
            ax = axes[row, col]
            for (cc_dir, dataset, label), color in zip(spaces, colors):
                try:
                    if metric == "neighborhood":
                        res = load_roc_artifact(cc_dir, dataset, kind)
                        fpr, tpr, auroc = res["fpr"], res["tpr"], float(res["auc"])
                    else:
                        curve = load_pair_validation_curve(cc_dir, dataset, kind)
                        fpr = curve["fpr"].to_numpy()
                        tpr = curve["tpr"].to_numpy()
                        auroc = float(np.trapz(tpr, fpr))
                except (FileNotFoundError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping %s/%s curve for %s: %s", kind, metric, dataset, exc
                    )
                    continue
                ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} ({auroc:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL_INK, lw=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            # The panel title is the only thing distinguishing the two
            # metrics -- the axes are FPR/TPR in all four panels -- so it
            # stays, trimmed to the two words that carry the distinction.
            ax.set_title(f"{kind.upper()} -- {metric}", fontsize=10)
            ax.legend(loc="lower right", fontsize=8, frameon=False)
            _clean_axes(ax)

    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_confidence_overlay(df: pd.DataFrame, output_path: Path) -> None:
    """Overlay the two spaces' sign3 confidence (applicability) distributions.

    Reproduces Fig. 3c for two spaces at once. Per Table S2 the confidence
    distribution is "space-dependent" and is used to rank compounds within a
    space, so this is read as a shift in the distribution rather than
    against any absolute pass mark.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compare_confidence`, tagged with display labels.
    output_path : Path
        Where to save the figure (PNG).
    """
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # unique() preserves first-appearance (A then B) order; groupby would
    # sort alphabetically and could silently swap which space gets which
    # color relative to every other figure.
    spaces = df["space"].unique().tolist()
    colors = _space_colors(len(spaces))
    for space, color in zip(spaces, colors):
        subset = df.loc[df["space"] == space, "confidence"]
        # The mean goes in the legend rather than as floating text at a
        # fraction of the current y-limit: that limit is still growing while
        # the histograms are drawn, so such a label can land off-axis.
        ax.hist(
            subset, bins=60, alpha=0.55, density=True, color=color,
            label=f"{space} (mean {subset.mean():.3f})",
        )
        ax.axvline(subset.mean(), color=color, linestyle="--", lw=1.5)
    ax.set_xlabel("sign3 confidence (applicability)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_across_roc_scatter(
    df: pd.DataFrame, output_path: Path, label_a: str, label_b: str
) -> None:
    """Compare the two spaces' per-CC-space recapitulation AUROCs (two panels).

    Left: each CC space as a point, space A on x and space B on y. Points
    below the y=x diagonal recapitulate better in space A. Whether the
    points fall on one side *systematically* is the real question -- a
    uniform shift across all 25 CC spaces means something quite different
    from a few large moves -- so the title reports the win/loss tally.

    Right: the same data as a sorted delta per CC space. The scatter alone
    is not enough here: with 25 points clustered along the diagonal, direct
    labels overlap badly and the ones that matter become unreadable, so the
    scatter labels only the largest movers and the delta panel carries the
    full, collision-free identity axis.

    Any CC space missing from one of the two diagnosis runs cannot be
    placed and is reported in the title rather than silently dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compare_across_roc`.
    output_path : Path
        Where to save the figure (PNG).
    label_a, label_b : str
        Display labels for the two compared spaces.
    """
    plt = _pyplot()

    complete = df.dropna(subset=["auroc_a", "auroc_b"]).copy()
    n_dropped = len(df) - len(complete)
    b_wins = int((complete["auroc_b"] > complete["auroc_a"]).sum())
    n = len(complete)

    fig, (ax_scatter, ax_delta) = plt.subplots(
        1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1, 1]}
    )

    # -- left: scatter, labelling only the biggest movers --------------------
    ax_scatter.scatter(
        complete["auroc_a"], complete["auroc_b"],
        s=42, color=SPACE_COLORS[0], edgecolor="white", linewidth=0.8, zorder=3,
    )
    movers = complete.reindex(
        complete["delta_b_minus_a"].abs().sort_values(ascending=False).index
    ).head(5)
    for _, row in movers.iterrows():
        ax_scatter.annotate(
            row["cc_space"],
            xy=(row["auroc_a"], row["auroc_b"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333330",
        )
    values = np.concatenate([complete["auroc_a"], complete["auroc_b"]])
    pad = 0.02
    lims = [float(values.min()) - pad, float(values.max()) + pad]
    ax_scatter.plot(lims, lims, linestyle="--", color=NEUTRAL_INK, lw=1, zorder=1)
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel(f"{label_a} AUROC")
    ax_scatter.set_ylabel(f"{label_b} AUROC")
    _clean_axes(ax_scatter)

    # -- right: sorted deltas, every space labelled --------------------------
    ordered = complete.sort_values("delta_b_minus_a")
    positions = np.arange(len(ordered))
    deltas = ordered["delta_b_minus_a"].to_numpy(dtype=float)
    # Color by direction, not by magnitude: which space wins is the question.
    colors = [SPACE_COLORS[1] if d > 0 else SPACE_COLORS[0] for d in deltas]
    ax_delta.barh(positions, deltas, color=colors, height=0.68)
    ax_delta.set_yticks(positions)
    ax_delta.set_yticklabels(ordered["cc_space"], fontsize=8)
    ax_delta.axvline(0, color=NEUTRAL_INK, lw=1)
    ax_delta.set_xlabel(f"AUROC difference ({label_b} - {label_a})")
    _clean_axes(ax_delta)
    for pos, delta in zip(positions, deltas):
        ax_delta.annotate(
            f"{delta:+.4f}",
            xy=(delta, pos),
            xytext=(-4 if delta < 0 else 4, 0),
            textcoords="offset points",
            ha="right" if delta < 0 else "left",
            va="center",
            fontsize=7,
            color="#333330",
        )
    xmin, xmax = ax_delta.get_xlim()
    span = xmax - xmin
    ax_delta.set_xlim(xmin - span * 0.22, xmax + span * 0.22)

    # Kept: the tally is a result, not a restatement of the axes.
    suptitle = f"{label_b} higher in {b_wins}/{n} CC spaces"
    if n_dropped:
        suptitle += f" ({n_dropped} space(s) missing from one run, not shown)"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


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

    For each cutoff in ``pvals``, plots the mean ROC curve across
    ``n_subsamples`` draws with a shaded +/- 1 std band, annotated
    ``pval:X.Xe-0Y - AUROC±std`` as in the paper.

    The two curves are two *cutoffs on one comparison*, not two spaces, so
    they take two steps of a single ramp rather than the A/B space colors --
    reusing those here would imply a space comparison that this figure is
    not drawing.

    Parameters
    ----------
    ref_vectors, query_vectors : numpy.ndarray
        Row-aligned vectors (see :func:`get_shared_vectors`).
    ref_label, query_label : str
        Display labels for the reference and query spaces.
    output_path : Path
        Where to save the figure (PNG).
    pvals : tuple of float, default (0.01, 0.001)
        Background-distribution cutoffs to plot, one curve each (the two
        cutoffs used in the CC Protocols paper).
    n_random, n_subsamples, random_state
        Forwarded to :func:`recapitulation_roc_band`.
    """
    plt = _pyplot()

    fpr_grid = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    colors = list(CUTOFF_COLORS)
    while len(colors) < len(pvals):
        colors.extend(CUTOFF_COLORS)

    for pval, color in zip(pvals, colors):
        try:
            band = recapitulation_roc_band(
                ref_vectors, query_vectors, pval, n_random, n_subsamples,
                random_state, fpr_grid,
            )
        except RuntimeError as exc:
            logger.warning("Skipping pval=%.4g curve: %s", pval, exc)
            continue
        ax.plot(
            fpr_grid, band["mean_tpr"], color=color, lw=2,
            label=f"pval:{pval:.1e} - {band['auroc']:.2f}±{band['std']:.2f}",
        )
        ax.fill_between(
            fpr_grid,
            band["mean_tpr"] - band["std_tpr"],
            band["mean_tpr"] + band["std_tpr"],
            color=color, alpha=0.2, linewidth=0,
        )
    # Styling below deliberately mirrors Extended Data Fig. 8g,h so the two can
    # sit side by side on a slide: green/blue cutoff ramp, dashed grey diagonal
    # and grid, quarter-step ticks, FPR/TPR axis labels and a boxed lower-right
    # legend reading ``pval:X.Xe-0Y - AUROC±std``.
    ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL_INK, lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(
        f"Recap. {ref_label} type III sign.\nusing {query_label} type III sign.",
        fontsize=11,
    )
    ax.grid(True, linestyle="--", color=NEUTRAL_INK, alpha=0.4, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=1.0)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NEUTRAL_INK)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_key_coverage_comparison(
    local_cc_dir_a: Path,
    dataset_a: str,
    label_a: str,
    local_cc_dir_b: Path,
    dataset_b: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Compare how many other CC spaces each space's compounds appear in.

    Table S2 ties this directly to type III signature confidence: the good
    outcome is most molecules present in at least 5-6 CC spaces, while a
    high proportion in fewer than 5 "typically indicat[es] little overlap
    with the CC universe (leading to unconfident type III signatures)". The
    dashed line marks that 5-space reading point.

    This is the most direct available check on whether newly added compounds
    connected to the rest of the CC universe or landed isolated.

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
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = _space_colors(2)
    entries = ((local_cc_dir_a, dataset_a, label_a), (local_cc_dir_b, dataset_b, label_b))
    for (cc_dir, dataset, label), color in zip(entries, colors):
        counts = np.asarray(
            load_diag_artifact(cc_dir, dataset, "key_coverage")["key_counts"], dtype=float
        )
        bins = np.arange(-0.5, 26.5, 1.0)
        ax.hist(
            counts, bins=bins, density=True, alpha=0.55, color=color,
            label=f"{label} (mean {counts.mean():.2f}, "
                  f"{np.mean(counts < 5) * 100:.1f}% in <5)",
        )
    # Table S2's reading point: below 5 CC spaces means weak overlap with
    # the CC universe. Marked with the line alone -- the legend already
    # carries the percentage each space has below it.
    ax.axvline(5, color=NEUTRAL_INK, linestyle="--", lw=1.2)
    ax.set_xlabel("Number of CC spaces the compound also appears in")
    ax.set_ylabel("Proportion of compounds")
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_structure_metrics_bar(extras_df: pd.DataFrame, output_path: Path) -> None:
    """Bar charts of the space-structure diagnosis metrics, with read direction.

    These are the Table S2 panels whose favorable direction is *not*
    uniformly "higher is better", so each subplot's title states which way
    it reads. Getting this wrong would invert the conclusion for two of the
    four metrics: more DBSCAN clusters is the good outcome (finer, more
    continuous structure rather than few redundant blobs), and lower CC
    ranks agreement is the good outcome (orthogonality to the rest of the CC
    rather than redundancy with it).

    Parameters
    ----------
    extras_df : pandas.DataFrame
        Output of :func:`build_diagnosis_extras_table`.
    output_path : Path
        Where to save the figure (PNG).

    Raises
    ------
    ValueError
        If none of the expected metric columns are present.
    """
    plt = _pyplot()

    # Each metric's favorable direction is documented on the function that
    # extracts it and in RESULTS.md, not printed on the panel: the reader of
    # a slide is being told the direction out loud, and "lower is better"
    # under every title is six lines of chrome competing with the values.
    specs = [
        ("pct_redundant", "Redundant signatures (%)", "{:.3f}"),
        ("n_clusters", "DBSCAN clusters", "{:.0f}"),
        ("pct_unclustered", "Unclustered / outlying (%)", "{:.2f}"),
        ("mean_ranks_agreement", "CC ranks agreement (RBO)", "{:.4f}"),
        ("mean_cc_spaces_per_compound", "Mean CC spaces per compound", "{:.3f}"),
        ("pct_in_fewer_than_5_spaces", "Compounds in <5 CC spaces (%)", "{:.2f}"),
    ]
    present = [(c, t, f) for c, t, f in specs if c in extras_df.columns]
    if not present:
        raise ValueError(
            f"No structure metrics found in extras_df (columns: {list(extras_df.columns)})."
        )

    ncols = min(3, len(present))
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    spaces = extras_df["space"].tolist()
    colors = _space_colors(len(spaces))
    for ax, (col, title, fmt) in zip(axes_flat, present):
        values = extras_df[col].to_numpy(dtype=float)
        bars = ax.bar(range(len(spaces)), values, color=colors, width=0.62)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9)
        _annotate_bars(ax, bars, values, fmt=fmt)
        _clean_axes(ax)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes_flat[len(present):]:
        ax.axis("off")

    _space_legend(fig, spaces, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig_atomic(fig, output_path)
    plt.close(fig)


def plot_signature_recovery_lollipop(
    recovery_df: pd.DataFrame, output_path: Path, pval: float = 0.01
) -> None:
    """Lollipop chart of within-space signature recovery (Fig. 3b style).

    Signature-type combinations run along the x axis with vertical stems, so
    the progression through the abstraction ladder reads left-to-right and
    the AUROC reads as height -- the direction people already associate with
    "more". The stems are anchored at the 0.5 chance baseline rather than at
    zero: for an AUROC the quantity of interest is the distance above chance,
    and a stem from zero gives every mark a long uninformative shaft that
    compresses the part that matters.

    A lollipop is used rather than paired bars because these values sit in a
    narrow band well above chance, where bar area is mostly dead space and
    the marker position is what the reader actually compares. The +/- 1 std
    across subsamples is drawn as a whisker through each marker.

    Parameters
    ----------
    recovery_df : pandas.DataFrame
        Concatenated output of :func:`signature_recovery` for both spaces.
    output_path : Path
        Where to save the figure (PNG).
    pval : float, default 0.01
        Cutoff the recovery was computed at, noted on the y-axis label.

    Raises
    ------
    ValueError
        If ``recovery_df`` is empty.
    """
    if recovery_df.empty:
        raise ValueError("recovery_df is empty; nothing to plot.")

    plt = _pyplot()

    spaces = recovery_df["space"].unique().tolist()
    colors = _space_colors(len(spaces))
    # Keep the SIGNATURE_PAIRS order (the abstraction ladder) rather than
    # whatever order the rows happen to arrive in.
    present = set(recovery_df["pair"])
    pair_order = [
        f"{lo}\u2192{hi}" for lo, hi in SIGNATURE_PAIRS
        if f"{lo}\u2192{hi}" in present
    ]
    positions = {pair: i for i, pair in enumerate(pair_order)}
    offsets = np.linspace(-0.16, 0.16, len(spaces)) if len(spaces) > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(1.55 * len(pair_order) + 2.6, 6.0))
    for space, color, offset in zip(spaces, colors, offsets):
        subset = recovery_df[recovery_df["space"] == space]
        for _, row in subset.iterrows():
            if row["pair"] not in positions:
                continue
            x = positions[row["pair"]] + offset
            ax.plot([x, x], [0.5, row["auroc"]], color=color, lw=2.4, solid_capstyle="butt")
            ax.errorbar(
                x, row["auroc"], yerr=row["std"], fmt="o", color=color,
                markersize=9, markeredgecolor="white", markeredgewidth=1.0,
                ecolor=color, elinewidth=1.4, capsize=3, zorder=3,
            )
            ax.annotate(
                f"{row['auroc']:.3f}",
                xy=(x, row["auroc"]), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=8, color="#333330",
            )
        # One legend entry per space, not one per lollipop.
        ax.plot([], [], "o-", color=color, lw=2.4, markersize=9, label=space)

    ax.axhline(0.5, color=NEUTRAL_INK, linestyle="--", lw=1.2)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=10)
    ax.set_xlim(-0.6, len(pair_order) - 0.4)
    ax.set_ylim(0.45, 1.04)
    ax.set_ylabel(f"NN recapitulation AUROC (p = {pval:g}; 0.5 = chance)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _clean_axes(ax)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


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
    """t-SNE overlay of genuine (sign0) input-compound membership.

    Mirrors Extended Data Fig. 2/5/8a: shows whether the compounds a dataset
    actually has raw data for populate previously uncovered regions of
    bioactivity space, or merely add density around already-covered ones.

    Membership comes from **sign0**, not sign3 -- sign3 is inferred for the
    whole CC universe, so a sign3-based overlay would mark everything as
    shared and show nothing. Both compound sets are projected through a
    single sign3 space so the embedding is self-consistent; two
    independently trained sign3 spaces have their own arbitrary orientations
    and could not be overlaid.

    "Present in both" is drawn first, in neutral ink, as context; the
    exclusive sets take the space colors and are drawn on top, since they
    are what the figure is actually about.

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
        Cap on total points projected, matching the ~10k cap CC's own
        diagnosis plots use.
    random_state : int, default 0
        Seed for subsampling and the t-SNE fit.

    Raises
    ------
    RuntimeError
        If fewer than 50 compounds are found in ``projection_sign3``.
    """
    plt = _pyplot()
    from sklearn.manifold import TSNE

    shared = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    groups = {
        "Present in both": sorted(shared),
        f"{label_a} only": sorted(only_a),
        f"{label_b} only": sorted(only_b),
    }
    all_keys = [k for members in groups.values() for k in members]
    membership = [g for g, members in groups.items() for _ in members]

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
    ordered = [membership_by_key[k] for k in found_keys]

    # perplexity must stay below the sample count; CC's own projections use
    # 30, but these input-compound sets are ~1e3, not ~1e4.
    perplexity = float(min(30, max(5, (len(found_keys) - 1) // 3)))
    embedding = TSNE(
        n_components=2, init="pca", random_state=random_state, perplexity=perplexity
    ).fit_transform(vectors)

    styles = {
        "Present in both": (NEUTRAL_INK, 10, 0.45, 1),
        f"{label_a} only": (SPACE_COLORS[0], 26, 0.9, 2),
        f"{label_b} only": (SPACE_COLORS[1], 26, 0.9, 3),
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    for group, (color, size, alpha, z) in styles.items():
        mask = np.array([m == group for m in ordered])
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=size, alpha=alpha, color=color, zorder=z, linewidth=0,
            label=f"{group} (n={int(mask.sum())})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Real (sign0) input compounds, shared sign3 projection", fontsize=11)
    ax.legend(markerscale=1.8, frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    _savefig_atomic(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _safe_plot(description: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run one plotting call, logging and swallowing any failure.

    A single bad figure should not cost the whole run -- the tables and the
    remaining figures are still worth producing. Failures are logged with a
    traceback so they stay visible rather than silently disappearing.

    Parameters
    ----------
    description : str
        Human-readable description used in the log messages.
    fn : callable
        The plotting function to call.
    *args, **kwargs
        Forwarded to ``fn``.
    """
    try:
        fn(*args, **kwargs)
        logger.info("Wrote %s", description)
    except Exception:
        logger.exception("Failed to write %s", description)


def _sanitize_filename(label: str) -> str:
    """Make a display label safe to embed in a filename.

    Spaces in output filenames are awkward to handle in shell pipelines and
    slide-deck asset folders, so they become underscores.

    Parameters
    ----------
    label : str
        Display label, e.g. ``"All Shared Proteins"``.

    Returns
    -------
    str
        Filename-safe form, e.g. ``"All_Shared_Proteins"``.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    return "_".join(filter(None, safe.split("_")))


def external_reference_recapitulation(
    cc: Any,
    reference_dataset: str,
    sign3_by_label: dict[str, Any],
    output_dir: Path,
    pvals: tuple[float, ...] = (0.01, 0.001),
    n_random: int = 2500,
    n_subsamples: int = 5,
    random_state: int | None = None,
    max_pool: int = 50000,
    make_plots: bool = True,
) -> pd.DataFrame | None:
    """Recapitulate an external CC space's neighbours with each compared space.

    This reproduces the exact design of Extended Data Fig. 8g,h, whose caption
    reads "Recapitulation of kNN **at X** type III signature level **using Y**
    type III signatures": ``reference_dataset`` plays the role of X (it defines
    the neighbour pairs) and each space in ``sign3_by_label`` plays the role of
    Y (it is scored on recovering them). The paper holds Y fixed at D6.001 and
    varies X; here X is held fixed and Y varies, so our two spaces land on the
    paper's own axis and can be read against its panel values directly.

    This is **not** the same measurement as the ``across_roc`` diagnosis
    artifact behind ``across_roc_scatter.png``, even though that also scores a
    space against D1.001. ``Diagnosis.cross_roc()`` defines positives as each
    molecule's ``k=5`` nearest neighbours and draws a *balanced* negative set;
    this test defines positives by a distance percentile (1% of pairs at
    P=0.01) and treats every other pair as negative. Different positive sets
    and very different class balance, so the two AUROCs are on different
    scales and must never be compared with each other.

    Parameters
    ----------
    cc : chemicalchecker.core.chemcheck.ChemicalChecker
        Instance holding the reference dataset's sign3.
    reference_dataset : str
        Dataset code whose neighbours define the positive pairs (e.g.
        ``"D1.001"``).
    sign3_by_label : dict of str to DataSignature
        Query spaces, keyed by display label.
    output_dir : Path
        Directory for the per-space figures and the summary CSV.
    pvals : tuple of float, default (0.01, 0.001)
        NN cutoffs, matching the paper's two curves.
    n_random, n_subsamples, random_state, max_pool
        Forwarded to :func:`get_shared_vectors` and
        :func:`recapitulation_roc_band`.
    make_plots : bool, default True
        Whether to write the ROC figures as well as the CSV.

    Returns
    -------
    pandas.DataFrame or None
        One row per (space, pval) with ``auroc``/``std``, or ``None`` if the
        reference signature could not be loaded.
    """
    try:
        ref_sign3 = cc.get_signature("sign3", "full", reference_dataset)
    except Exception as exc:  # noqa: BLE001 -- CC raises bare Exception here
        logger.warning(
            "Skipping external-reference recapitulation against %s: %s",
            reference_dataset, exc,
        )
        return None

    rows: list[dict[str, Any]] = []
    for label, query_sign3 in sign3_by_label.items():
        _, ref_vec, query_vec = get_shared_vectors(
            ref_sign3, query_sign3, max_pool=max_pool, random_state=random_state
        )
        for pval in pvals:
            try:
                band = recapitulation_roc_band(
                    ref_vec, query_vec, pval, n_random, n_subsamples,
                    random_state, np.linspace(0, 1, 200),
                )
            except RuntimeError as exc:
                logger.warning(
                    "%s vs %s at pval=%.4g: %s", reference_dataset, label, pval, exc
                )
                continue
            rows.append({
                "reference_dataset": reference_dataset,
                "query_space": label,
                "pval": pval,
                "auroc": band["auroc"],
                "std": band["std"],
                "n_shared_compounds": int(ref_vec.shape[0]),
            })
            logger.info(
                "%s neighbours recapitulated by %s at pval=%.0e: "
                "AUROC=%.3f +/- %.3f",
                reference_dataset, label, pval, band["auroc"], band["std"],
            )
        if make_plots:
            name = (
                f"recap_roc_{_sanitize_filename(reference_dataset)}"
                f"_by_{_sanitize_filename(label)}.png"
            )
            _safe_plot(
                name,
                plot_recapitulation_roc,
                ref_vec, query_vec, reference_dataset, label, output_dir / name,
                pvals=pvals, n_random=n_random, n_subsamples=n_subsamples,
                random_state=random_state,
            )

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "external_reference_recapitulation.csv", index=False)
    return df


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
    max_pool: int = 50000,
    reference_datasets: tuple[str, ...] = (),
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run the full space-vs-space comparison and write outputs to disk.

    Produces three complementary layers of evidence, which answer different
    questions and should not be collapsed into one another:

    1. **Whole-universe diagnosis metrics** (``summary_table.csv``,
       ``diagnosis_extras.csv``) -- computed over each space's entire
       inferred population, so the two spaces are not compared at equal
       difficulty.
    1b. **Within-space signature recovery** (``signature_recovery.csv``,
       ``signature_recovery_lollipop.png``) -- the Fig. 3b check that each
       space preserves its own raw-data structure as it is abstracted from
       sign0 up to sign3, restricted to that space's real input compounds.
    2. **Cross-space NN recapitulation** (``shared_key_recapitulation.json``,
       ``recap_roc_*.png``) -- how much of each space's neighbor structure
       the other reproduces, over the shared key set.
    3. **Real input-compound membership** (``sign0_membership.json``,
       ``input_compound_tsne_overlay.png``) -- what each dataset actually
       measured, which is the only layer where "more data" is literally true.

    Parameters
    ----------
    local_cc_dir_a, local_cc_dir_b : Path
        Local CC instance directories for space A and space B (may be the
        same directory if both dataset codes live in one instance).
    dataset_a, dataset_b : str
        CC dataset codes for space A and space B.
    label_a, label_b : str
        Display labels used in tables and plot legends.
    output_dir : Path
        Directory to write comparison tables/figures to (created if missing).
    pval : float, default 0.01
        Background-distribution p-value defining the NN cutoff.
    n_random : int, default 2500
        Compounds subsampled per recapitulation repetition.
    n_subsamples : int, default 5
        Recapitulation repetitions to average over.
    random_state : int, optional
        Seed for reproducible subsampling.
    max_pool : int, default 50000
        Cap on shared compounds fetched for the recapitulation test; ``0``
        fetches all of them (see :func:`get_shared_vectors`).
    reference_datasets : tuple of str, default ()
        External CC dataset codes (e.g. ``("D1.001",)``) whose sign3
        neighbours each compared space is scored on recovering, in the design
        of Extended Data Fig. 8g,h. See
        :func:`external_reference_recapitulation`.
    make_plots : bool, default True
        Whether to save comparison figures in addition to tables.

    Returns
    -------
    dict
        ``summary_table``, ``diagnosis_extras``, ``across_roc_comparison``,
        ``sign0_membership`` and ``shared_key_recapitulation``. Figures are
        written to ``output_dir`` rather than returned; each figure's
        failure is logged and skipped independently.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # CC_CONFIG must already be set by the caller before importing chemicalchecker.
    from chemicalchecker import ChemicalChecker

    logger.info("Loading sign3 signatures for %s and %s ...", label_a, label_b)
    cc_a = ChemicalChecker(str(local_cc_dir_a), dbconnect=False)
    cc_b = cc_a if local_cc_dir_a == local_cc_dir_b else ChemicalChecker(
        str(local_cc_dir_b), dbconnect=False
    )
    sign3_a = cc_a.get_signature("sign3", "full", dataset_a)
    sign3_b = cc_b.get_signature("sign3", "full", dataset_b)

    specs = [SpaceSpec(label_a, dataset_a), SpaceSpec(label_b, dataset_b)]
    local_cc_dirs = {label_a: local_cc_dir_a, label_b: local_cc_dir_b}

    # --- Layer 1: whole-universe diagnosis metrics --------------------------
    logger.info("Building summary table from saved diagnosis artifacts ...")
    summary_df = build_summary_table(specs, local_cc_dirs)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    results["summary_table"] = summary_df
    for kind in ("moa", "atc"):
        neigh = summary_df[f"{kind}_auroc_neighborhood"].tolist()
        annot = summary_df[f"{kind}_auroc_annotation"].tolist()
        logger.info(
            "%s AUROC -- neighborhood (canvas): %s=%.4f vs %s=%.4f | "
            "annotation (validation_stats.json): %s=%.4f vs %s=%.4f",
            kind.upper(), label_a, neigh[0], label_b, neigh[1],
            label_a, annot[0], label_b, annot[1],
        )
        if (neigh[1] > neigh[0]) != (annot[1] > annot[0]):
            logger.warning(
                "%s: the two validation metrics disagree about which space is "
                "better. Both are reported; the neighborhood metric is the "
                "paper-faithful/canvas one.", kind.upper(),
            )

    logger.info("Building diagnosis-extras table ...")
    extras_df = build_diagnosis_extras_table(specs, local_cc_dirs)
    extras_df.to_csv(output_dir / "diagnosis_extras.csv", index=False)
    results["diagnosis_extras"] = extras_df

    logger.info("Comparing per-CC-space NN recapitulation (across_roc) ...")
    across_df = None
    try:
        across_df = compare_across_roc(local_cc_dir_a, dataset_a, local_cc_dir_b, dataset_b)
        across_df.to_csv(output_dir / "across_roc_comparison.csv", index=False)
        results["across_roc_comparison"] = across_df
        complete = across_df.dropna(subset=["auroc_a", "auroc_b"])
        logger.info(
            "%s has the higher AUROC in %d/%d CC spaces (mean delta B-A: %+.4f).",
            label_b,
            int((complete["auroc_b"] > complete["auroc_a"]).sum()),
            len(complete),
            complete["delta_b_minus_a"].mean(),
        )
    except (FileNotFoundError, TypeError) as exc:
        logger.warning("Skipping across_roc comparison: %s", exc)

    # --- Layer 3 (tables): real input-compound membership -------------------
    logger.info("Comparing real (sign0) input compounds ...")
    membership = None
    try:
        membership = sign0_membership_summary(
            cc_a, dataset_a, label_a, cc_b, dataset_b, label_b
        )
        with open(output_dir / "sign0_membership.json", "w", encoding="utf-8") as fh:
            json.dump(membership, fh, indent=2)
        results["sign0_membership"] = membership
        logger.info(
            "Input compounds: %s=%d (%d features), %s=%d (%d features); "
            "shared=%d, only-%s=%d, only-%s=%d",
            label_a, membership["n_input_compounds_a"], membership["n_features_a"],
            label_b, membership["n_input_compounds_b"], membership["n_features_b"],
            membership["n_shared_input_compounds"],
            label_a, membership["n_only_a"], label_b, membership["n_only_b"],
        )
        if membership["a_is_subset_of_b"]:
            logger.info(
                "%s's input compounds are a strict subset of %s's -- the "
                "extension is purely additive in compounds.", label_a, label_b,
            )
        if membership["n_features_b"] < membership["n_features_a"]:
            logger.warning(
                "%s measures FEWER features than %s (%d vs %d). The extension "
                "trades measured features for compounds; read every downstream "
                "metric in that light.",
                label_b, label_a,
                membership["n_features_b"], membership["n_features_a"],
            )
    except Exception:
        logger.exception("Failed to summarize sign0 membership")

    # --- Layer 1b: within-space signature recovery (Fig. 3b) ----------------
    logger.info("Computing within-space signature recovery (sign0 -> sign1/2/3 ...) ...")
    recovery_df = None
    try:
        recovery_df = pd.concat(
            [
                signature_recovery(
                    cc, code, lab, pval=pval, n_random=n_random,
                    n_subsamples=n_subsamples, random_state=random_state,
                )
                for cc, code, lab in (
                    (cc_a, dataset_a, label_a),
                    (cc_b, dataset_b, label_b),
                )
            ],
            ignore_index=True,
        )
        recovery_df.to_csv(output_dir / "signature_recovery.csv", index=False)
        results["signature_recovery"] = recovery_df
    except Exception:
        logger.exception("Failed to compute signature recovery")

    # --- Layer 2: cross-space NN recapitulation -----------------------------
    logger.info(
        "Running shared-compound NN recapitulation (pval=%.4g, n_random=%d, "
        "n_subsamples=%d, max_pool=%d) ...",
        pval, n_random, n_subsamples, max_pool,
    )
    n_shared_total = len(set(sign3_a.keys) & set(sign3_b.keys))
    _, vec_a, vec_b = get_shared_vectors(
        sign3_a, sign3_b, max_pool=max_pool, random_state=random_state
    )
    recap = shared_key_recapitulation(
        vec_a, vec_b, n_shared_compounds=n_shared_total, pval=pval,
        n_random=n_random, n_subsamples=n_subsamples, random_state=random_state,
    )
    with open(output_dir / "shared_key_recapitulation.json", "w", encoding="utf-8") as fh:
        json.dump(recap, fh, indent=2)
    results["shared_key_recapitulation"] = recap
    logger.info(
        "%s NN recapitulated by %s: AUROC=%.3f +/- %.3f "
        "(cutoff=%.5f, positive rate=%.4f)",
        label_a, label_b,
        recap["a_recap_by_b"]["auroc"], recap["a_recap_by_b"]["std"],
        recap["a_recap_by_b"]["cutoff"], recap["a_recap_by_b"]["mean_positive_rate"],
    )
    logger.info(
        "%s NN recapitulated by %s: AUROC=%.3f +/- %.3f "
        "(cutoff=%.5f, positive rate=%.4f)",
        label_b, label_a,
        recap["b_recap_by_a"]["auroc"], recap["b_recap_by_a"]["std"],
        recap["b_recap_by_a"]["cutoff"], recap["b_recap_by_a"]["mean_positive_rate"],
    )

    if not make_plots:
        return results

    # --- Figures ------------------------------------------------------------
    _safe_plot(
        "headline_metrics_bar.png",
        plot_headline_metrics_bar, summary_df, output_dir / "headline_metrics_bar.png",
    )
    _safe_plot(
        "moa_atc_roc_overlay.png",
        plot_moa_atc_roc_overlay,
        local_cc_dir_a, dataset_a, label_a, local_cc_dir_b, dataset_b, label_b,
        output_dir / "moa_atc_roc_overlay.png",
    )
    if recovery_df is not None:
        _safe_plot(
            "signature_recovery_lollipop.png",
            plot_signature_recovery_lollipop,
            recovery_df, output_dir / "signature_recovery_lollipop.png", pval,
        )
    _safe_plot(
        "structure_metrics_bar.png",
        plot_structure_metrics_bar, extras_df, output_dir / "structure_metrics_bar.png",
    )
    _safe_plot(
        "key_coverage_comparison.png",
        plot_key_coverage_comparison,
        local_cc_dir_a, dataset_a, label_a, local_cc_dir_b, dataset_b, label_b,
        output_dir / "key_coverage_comparison.png",
    )
    if across_df is not None:
        _safe_plot(
            "across_roc_scatter.png",
            plot_across_roc_scatter,
            across_df, output_dir / "across_roc_scatter.png", label_a, label_b,
        )
    try:
        conf_df = compare_confidence(
            local_cc_dir_a, dataset_a, label_a, local_cc_dir_b, dataset_b, label_b
        )
        _safe_plot(
            "confidence_overlay.png",
            plot_confidence_overlay, conf_df, output_dir / "confidence_overlay.png",
        )
    except FileNotFoundError as exc:
        logger.warning("Skipping confidence comparison: %s", exc)

    if membership is not None:
        # Project through whichever sign3 space covers more of the CC
        # universe, so the fewest input compounds fall outside the embedding.
        projection_sign3, projection_label = (
            (sign3_a, label_a) if len(sign3_a.keys) >= len(sign3_b.keys) else (sign3_b, label_b)
        )
        logger.info("Projecting input-compound overlay using %s's sign3 space.", projection_label)
        _safe_plot(
            "input_compound_tsne_overlay.png",
            plot_input_compound_overlay,
            projection_sign3,
            get_sign0_keys(cc_a, dataset_a),
            get_sign0_keys(cc_b, dataset_b),
            label_a, label_b,
            output_dir / "input_compound_tsne_overlay.png",
        )

    for ref_vec, query_vec, ref_label, query_label in (
        (vec_a, vec_b, label_a, label_b),
        (vec_b, vec_a, label_b, label_a),
    ):
        name = f"recap_roc_{_sanitize_filename(ref_label)}_NN.png"
        _safe_plot(
            name,
            plot_recapitulation_roc,
            ref_vec, query_vec, ref_label, query_label, output_dir / name,
            n_random=n_random, n_subsamples=n_subsamples, random_state=random_state,
        )

    for reference_dataset in reference_datasets:
        logger.info(
            "Recapitulating %s neighbours with each compared space "
            "(Ext. Data Fig. 8g,h design) ...", reference_dataset,
        )
        ext_df = external_reference_recapitulation(
            cc_a, reference_dataset,
            {label_a: sign3_a, label_b: sign3_b},
            output_dir,
            n_random=n_random, n_subsamples=n_subsamples,
            random_state=random_state, max_pool=max_pool, make_plots=make_plots,
        )
        if ext_df is not None:
            results.setdefault("external_reference_recapitulation", {})[
                reference_dataset
            ] = ext_df

    return results
