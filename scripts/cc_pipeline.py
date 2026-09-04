"""
Shared helper functions for the Chemical Checker "PerturbProt" signature pipeline.

Covers the sign0 -> sign1/neig1 -> sign2 -> sign3 stages. Both the
non-interactive CLI script (``pertprot_cli.py``) and the interactive
jupytext notebook (``pertprot_interactive.py``) import from this module,
so the actual fitting/diagnosis logic lives in exactly one place.

Chemical Checker objects (``cc_local`` and every ``signN``/``neig1``/``diagN``
object) come from the external ``chemicalchecker`` package, which ships no
type stubs; they are type-hinted as ``Any`` throughout and described in each
docstring instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils import read_table_with_required_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """
    Everything that differs from one PerturbProt dataset to another.

    Parameters
    ----------
    key : str
        Short identifier, e.g. ``"dcmoa"``. Used in logs and as a dict key.
    name : str
        Display name, e.g. ``"DeepCoverMoa"``.
    code : str
        Chemical Checker D6 dataset code, e.g. ``"D6.002"``.
    csv_path : str or os.PathLike
        Path to the wide CSV (rows=compounds keyed by InChIKey, columns=UniProt ids).
    df : pd.DataFrame or None
        The loaded dataframe, populated by :func:`load_dataset_dataframes`.
        ``None`` until then.
    """
    key: str
    name: str
    code: str
    csv_path: str | os.PathLike[str]
    df: pd.DataFrame | None = field(default=None, repr=False)


# 6 PerturbProt datasets: (key, display name, D6 code, csv filename).
# Single source of truth for both default_dataset_specs() and CLI --datasets validation.
_DATASET_DEFINITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("dcmoa", "DeepCoverMoa", "D6.002", "deepcovermoa_wide.csv"),
    ("dcmoa_deps", "DeepCoverMoa DEPs", "D6.003", "deepcovermoa_deps_wide.csv"),
    ("deps", "DEPs", "D6.004", "all_studies_deps_wide.csv"),
    ("shared_deps", "Shared DEPs", "D6.005", "all_studies_deps_wide_shared.csv"),
    ("allprot", "All Proteins", "D6.006", "all_studies_wide.csv"),
    ("shared_allprot", "All Shared Proteins", "D6.007", "all_studies_wide_shared.csv"),
)

DATASET_KEYS: tuple[str, ...] = tuple(definition[0] for definition in _DATASET_DEFINITIONS)


def default_dataset_specs(data_dir: str | os.PathLike[str]) -> list[DatasetSpec]:
    """
    Build the 6 standard PerturbProt DatasetSpecs for a given data directory.

    Parameters
    ----------
    data_dir : str or os.PathLike
        Base directory containing the wide compound-matrix CSVs (as produced
        by ``transform_allstudies_to_widedf.py``), equivalent to
        ``/scratch/sbnb/sayala/protocols/data/PerturbProt`` in the original script.

    Returns
    -------
    list of DatasetSpec
        One spec per dataset, in the order dcmoa, dcmoa_deps, deps,
        shared_deps, allprot, shared_allprot.
    """
    data_dir = Path(data_dir)
    return [
        DatasetSpec(key=key, name=name, code=code, csv_path=data_dir / fname)
        for key, name, code, fname in _DATASET_DEFINITIONS
    ]


def load_dataset_dataframes(specs: Sequence[DatasetSpec]) -> Sequence[DatasetSpec]:
    """
    Load the wide CSV for every spec into ``spec.df``, in place.

    Delegates existence-checking and logging to
    :func:`utils.read_table_with_required_columns`. No fixed column list is
    enforced since columns are dynamic UniProt ids; only the InChIKey index
    column is required implicitly via ``index_col=0``.

    Parameters
    ----------
    specs : sequence of DatasetSpec
        Specs to populate. Mutated in place.

    Returns
    -------
    sequence of DatasetSpec
        The same ``specs``, returned for chaining.

    Raises
    ------
    FileNotFoundError
        If any spec's ``csv_path`` does not exist.
    """
    for spec in specs:
        spec.df = read_table_with_required_columns(
            spec.csv_path, columns=[], sep=",", index_col=0
        )
        logger.info("[%s] dataframe ready: shape=%s", spec.key, spec.df.shape)
    return specs


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def report_minmax(sign_obj: Any, label: str | None = None) -> tuple[float, float]:
    """
    Log and return the (min, max) of a signature's underlying array.

    Replaces the repeated ``np.min(np.array(sign).flatten()), np.max(...)``
    line that followed every fit in the original script.

    Parameters
    ----------
    sign_obj : chemicalchecker signature object
        Any fitted sign0/sign1/sign2/sign3 object; must support ``np.array()``.
    label : str, optional
        Prefix used in the log message, e.g. ``"DeepCoverMoa sign2"``.

    Returns
    -------
    tuple of float
        ``(min, max)`` of the flattened array.
    """
    arr = np.array(sign_obj)
    vmin, vmax = np.min(arr), np.max(arr)
    logger.info("%sshape=%s min=%s max=%s", f"[{label}] " if label else "", arr.shape, vmin, vmax)
    return vmin, vmax


def diagnose_and_plot(
    sign_obj: Any,
    sizes: Sequence[str] = ("small",),
    ref_cctype: str | None = None,
    dpi: int = 300,
) -> Any:
    """
    Run ``.diagnosis()`` then ``.canvas()`` for one or more plot sizes.

    Replaces the repeated
    ``diagN.canvas(size=..., savefig=True, savefig_kwargs={'dpi': 300})``
    block that followed every fit in the original script.

    Parameters
    ----------
    sign_obj : chemicalchecker signature object
        The signature to diagnose.
    sizes : sequence of str, default ("small",)
        Canvas sizes to plot and save, e.g. ``("medium", "small")``.
    ref_cctype : str, optional
        Reference signature type forwarded to ``.diagnosis()`` (used for
        sign3, which diagnoses against ``ref_cctype='sign3'``).
    dpi : int, default 300
        Resolution used when saving each canvas.

    Returns
    -------
    chemicalchecker diagnosis object
        The object returned by ``sign_obj.diagnosis()``.
    """
    diag_kwargs = {} if ref_cctype is None else {"ref_cctype": ref_cctype}
    diag = sign_obj.diagnosis(**diag_kwargs)
    for size in sizes:
        diag.canvas(size=size, savefig=True, savefig_kwargs={"dpi": dpi})
    return diag


def get_cc_universe(cc_local: Any, exclude_code: str = "D6.001") -> set:
    """
    Compute the union of molecule keys across the canonical CC sign2 spaces.

    This is dataset-independent, so it should be computed ONCE per run and
    reused for every PerturbProt dataset (the original script re-derived the
    equivalent set inline for each dataset separately).

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    exclude_code : str, default "D6.001"
        Dataset code to exclude from the universe (the PerturbProt reference
        space itself).

    Returns
    -------
    set
        Molecule keys present in any canonical (``*.001``) CC sign2 'full' space.
    """
    universe: list = []
    for dat in cc_local.datasets:
        if dat.endswith("001") and dat != exclude_code:
            universe.extend(cc_local.get_signature("sign2", "full", dat).keys)
    universe = set(universe)
    logger.info("Number of molecules in the CC universe: %d", len(universe))
    return universe


def report_universe_overlap(name: str, sign2_obj: Any, cc_universe: set) -> set:
    """
    Log molecule-count and CC-universe overlap for one dataset's sign2.

    Parameters
    ----------
    name : str
        Display name of the dataset, used in the log message.
    sign2_obj : chemicalchecker signature object
        The dataset's fitted sign2.
    cc_universe : set
        Molecule keys from :func:`get_cc_universe`.

    Returns
    -------
    set
        Molecule keys present in ``sign2_obj``.
    """
    d6_molecules = set(sign2_obj.keys)
    overlap = len(cc_universe.intersection(d6_molecules))
    logger.info(
        "[%s] molecules in D6 sign2: %d | intersection with CC universe: %d",
        name, len(d6_molecules), overlap,
    )
    return d6_molecules


# ---------------------------------------------------------------------------
# Per-stage fitting functions
# ---------------------------------------------------------------------------

def fit_sign0(
    cc_local: Any, code: str, df: pd.DataFrame, sanitizer_kwargs: dict[str, Any] | None = None
) -> Any:
    """
    Instantiate, clear and fit a sign0 from a wide compound x feature dataframe.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    code : str
        D6 dataset code, e.g. "D6.002".
    df : pd.DataFrame
        Wide dataframe: rows are compounds (InChIKey index), columns are
        UniProt ids.
    sanitizer_kwargs : dict, optional
        Forwarded to ``sign0.fit()``, e.g. ``{"chunk_size": 500_000}``.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign0.
    """
    sign0 = cc_local.signature(code, "sign0")
    sign0.clear_all()  # cleaning both full and reference datasets -- crucial!
    sign0.fit(
        X=df.values,
        keys=list(df.index),
        features=list(df.columns),
        sanitizer_kwargs=sanitizer_kwargs or {},
    )
    return sign0


def fit_sign1(cc_local: Any, code: str, sign0: Any) -> tuple[Any, Any]:
    """
    Instantiate, clear and fit sign1 from sign0, plus the paired neig1.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    code : str
        D6 dataset code.
    sign0 : chemicalchecker signature object
        The dataset's fitted sign0.

    Returns
    -------
    tuple of (sign1, neig1)
        The fitted sign1 and its paired neig1.
    """
    sign1 = cc_local.signature(code, "sign1")
    sign1.clear_all()
    sign1.fit(sign0)

    neig1 = cc_local.get_signature("neig1", "full", code)  # takes the reference anyway
    neig1.clear_all()
    neig1.fit(sign1)

    return sign1, neig1


def fit_sign2(cc_local: Any, code: str, sign1: Any, neig1: Any, oos_predictor: bool = False) -> Any:
    """
    Instantiate, clear and fit sign2 from sign1 + neig1.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    code : str
        D6 dataset code.
    sign1 : chemicalchecker signature object
        The dataset's fitted sign1.
    neig1 : chemicalchecker signature object
        The dataset's fitted neig1.
    oos_predictor : bool, default False
        Forwarded to ``sign2.fit()``.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign2.
    """
    sign2 = cc_local.signature(code, "sign2")
    sign2.clear_all()
    sign2.fit(sign1, neig1, oos_predictor=oos_predictor)
    return sign2


def build_sign3_sign2_list(cc_local: Any, code: str) -> list[Any]:
    """
    Build the list of sign2 spaces that feed sign3.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    code : str
        D6 dataset code whose own sign2 'full' is appended last.

    Returns
    -------
    list
        The canonical CC sign2 'full' spaces (one per ``cc_local.coordinates``
        entry, i.e. the ~25 canonical spaces) plus this dataset's own sign2
        'full' -- 26 spaces total.
    """
    sign2_list = []
    for ds in cc_local.coordinates:
        ds_code = ds + ".001"
        sign2_list.append(cc_local.get_signature("sign2", "full", ds_code))
    sign2_list.append(cc_local.get_signature("sign2", "full", code))
    return sign2_list


def fit_sign3(
    cc_local: Any,
    code: str,
    sign2: Any,
    sign1: Any,
    mapping_dict: dict[str, str] | None = None,
    sign2_universe: Any = None,
    complete_universe: str = "fast",
    sign2_coverage: Any = None,
) -> Any:
    """
    Instantiate, clear and fit sign3 given sign2 + sign1, and the 26-space sign2 list.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    code : str
        D6 dataset code.
    sign2 : chemicalchecker signature object
        The dataset's fitted sign2.
    sign1 : chemicalchecker signature object
        The dataset's fitted sign1.
    mapping_dict : dict, optional
        In-house InChIKey -> InChI mapping, forwarded to ``sign3.fit()``.
    sign2_universe, complete_universe, sign2_coverage
        Forwarded to ``sign3.fit()`` unchanged; see chemicalchecker docs.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign3.

    Notes
    -----
    CAUTION: computationally demanding step -- normally run on an HPC cluster.
    """
    sign3 = cc_local.signature(code, "sign3")

    sign2_list = build_sign3_sign2_list(cc_local, code)
    logger.info("[%s] number of sign2 spaces feeding sign3: %d", code, len(sign2_list))

    sign3.clear_all()
    sign3.fit(
        sign2_list, sign2, sign1,
        sign2_universe=sign2_universe,
        complete_universe=complete_universe,
        sign2_coverage=sign2_coverage,
        dbconnect=False,
        mapping_dict=mapping_dict,
    )
    return sign3


# ---------------------------------------------------------------------------
# Full per-dataset orchestration (used by the non-interactive HPC script)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    cc_local: Any,
    spec: DatasetSpec,
    sanitizer_kwargs: dict[str, Any] | None = None,
    mapping_dict: dict[str, str] | None = None,
    cc_universe: set | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """
    Run sign0 -> sign1(+neig1) -> sign2 -> sign3 for one dataset spec.

    Parameters
    ----------
    cc_local : chemicalchecker.ChemicalChecker
        The local Chemical Checker instance.
    spec : DatasetSpec
        The dataset to process (``spec.df`` must already be loaded).
    sanitizer_kwargs : dict, optional
        Forwarded to :func:`fit_sign0`.
    mapping_dict : dict, optional
        Forwarded to :func:`fit_sign3`.
    cc_universe : set, optional
        If given, logs this dataset's overlap with the CC universe (see
        :func:`get_cc_universe`) before fitting sign3.
    plot : bool, default True
        Whether to run diagnosis/canvas plotting after every stage.

    Returns
    -------
    dict
        Every fitted signature object for this dataset, keyed by stage name:
        ``'sign0'``, ``'sign1'``, ``'neig1'``, ``'sign2'``, ``'sign3'``.
    """
    results: dict[str, Any] = {}

    sign0 = fit_sign0(cc_local, spec.code, spec.df, sanitizer_kwargs)
    if plot:
        diagnose_and_plot(sign0)
    report_minmax(sign0, label=f"{spec.name} sign0")
    results["sign0"] = sign0

    sign1, neig1 = fit_sign1(cc_local, spec.code, sign0)
    if plot:
        diagnose_and_plot(sign1)
    report_minmax(sign1, label=f"{spec.name} sign1")
    results["sign1"], results["neig1"] = sign1, neig1

    sign2 = fit_sign2(cc_local, spec.code, sign1, neig1)
    if plot:
        diagnose_and_plot(sign2)
    report_minmax(sign2, label=f"{spec.name} sign2")
    results["sign2"] = sign2

    if cc_universe is not None:
        report_universe_overlap(spec.name, sign2, cc_universe)

    sign3 = fit_sign3(cc_local, spec.code, sign2, sign1, mapping_dict=mapping_dict)
    if plot:
        diagnose_and_plot(sign3, sizes=("medium", "small"), ref_cctype="sign3")
    report_minmax(sign3, label=f"{spec.name} sign3")
    results["sign3"] = sign3

    return results
