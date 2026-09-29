"""Fitting of Chemical Checker signatures sign0 -> sign1/neig1 -> sign2 -> sign3 for one dataset.

Each ``fit_signN`` function is a thin wrapper around the chemicalchecker
``fit()`` of that signature type (it clears previous results first), and
:func:`run_signature_pipeline` chains them for one dataset as in the paper's
Procedure (Comajuncosa-Creus et al., Nat. Protoc. 2025).

chemicalchecker objects ship no type stubs, so they are annotated as ``Any``
and described in each docstring.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import (
    PIPELINE_STAGES,
    DatasetConfig,
    ExtendedSpace,
    PipelineStage,
    check_stage_range,
)
from .data_loaders import build_sign0_inputs

logger = logging.getLogger(__name__)


def report_minmax(signature: Any, label: str | None = None) -> tuple[float, float]:
    """
    Log and return the (min, max) of a signature's matrix.

    Parameters
    ----------
    signature : chemicalchecker signature object
        Any fitted sign0/sign1/sign2/sign3; must support ``np.array()``.
    label : str, optional
        Prefix used in the log message, e.g. ``"M1.001 sign2"``.

    Returns
    -------
    tuple of float
        ``(min, max)`` of the flattened matrix.
    """
    matrix = np.array(signature)
    value_min, value_max = np.min(matrix), np.max(matrix)
    logger.info(
        "%sshape=%s min=%s max=%s",
        f"[{label}] " if label else "",
        matrix.shape,
        value_min,
        value_max,
    )
    return value_min, value_max


def diagnose_and_plot(
    signature: Any,
    sizes: Sequence[str] = ("small",),
    ref_cctype: str | None = None,
    dpi: int = 300,
) -> Any:
    """
    Run the CC diagnosis of a signature and save its canvas at one or more sizes.

    Parameters
    ----------
    signature : chemicalchecker signature object
        The signature to diagnose.
    sizes : sequence of str, default ("small",)
        Canvas sizes to plot and save, e.g. ``("medium", "small")``.
    ref_cctype : str, optional
        Reference signature type forwarded to ``.diagnosis()`` (sign3 is
        diagnosed against ``ref_cctype="sign3"``).
    dpi : int, default 300
        Resolution used when saving each canvas.

    Returns
    -------
    chemicalchecker diagnosis object
        The object returned by ``signature.diagnosis()``.
    """
    diagnosis_kwargs = {} if ref_cctype is None else {"ref_cctype": ref_cctype}
    diagnosis = signature.diagnosis(**diagnosis_kwargs)
    for size in sizes:
        diagnosis.canvas(size=size, savefig=True, savefig_kwargs={"dpi": dpi})
    return diagnosis


def get_cc_universe(cc_instance: Any) -> set[str]:
    """
    Union of the InChIKeys in the sign2 of the 25 exemplary CC spaces.

    Uses ``ChemicalChecker.datasets_exemplary()`` rather than
    ``ChemicalChecker.universe``, which queries the CC database and does not
    work in local instances opened with ``dbconnect=False``. Compute it once
    per run and reuse it for every dataset.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.

    Returns
    -------
    set of str
        InChIKeys present in any exemplary sign2 space. Spaces whose sign2 is
        missing are skipped with a warning.
    """
    universe: set[str] = set()
    for dataset_code in cc_instance.datasets_exemplary():
        sign2 = cc_instance.get_signature("sign2", "full", dataset_code)
        if not sign2.available():
            logger.warning(
                "No sign2 for exemplary space %s; left out of the CC universe",
                dataset_code,
            )
            continue
        universe.update(sign2.keys)
    logger.info("Number of molecules in the CC universe: %d", len(universe))
    return universe


def report_universe_overlap(
    dataset_name: str, sign2: Any, cc_universe: set[str]
) -> set[str]:
    """
    Log how many of a dataset's sign2 molecules are already in the CC universe.

    Poor overlap makes sign3 unreliable (paper Procedure step 20).

    Parameters
    ----------
    dataset_name : str
        Display name used in the log message.
    sign2 : chemicalchecker signature object
        The dataset's fitted sign2.
    cc_universe : set of str
        InChIKeys from :func:`get_cc_universe`.

    Returns
    -------
    set of str
        InChIKeys present in ``sign2``.
    """
    dataset_inchikeys = set(sign2.keys)
    overlap = len(cc_universe & dataset_inchikeys)
    logger.info(
        "[%s] molecules in sign2: %d | intersection with CC universe: %d",
        dataset_name,
        len(dataset_inchikeys),
        overlap,
    )
    return dataset_inchikeys


def fit_sign0(
    cc_instance: Any,
    dataset_code: str,
    sign0_inputs: dict[str, Any],
    **fit_options: Any,
) -> Any:
    """
    Clear and fit a dataset's sign0.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        CC dataset code, e.g. ``"M1.001"``.
    sign0_inputs : dict
        Data arguments from :func:`~chemcheck_protocols.data_loaders.build_sign0_inputs`
        (``X``/``keys``/``features``, ``pairs`` or ``data_file``).
    **fit_options
        Forwarded to ``sign0.fit``, e.g. ``sanitizer_kwargs``.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign0.
    """
    sign0 = cc_instance.signature(dataset_code, "sign0")
    sign0.clear_all()  # clears both the full and the reference molsets
    sign0.fit(**sign0_inputs, **fit_options)
    return sign0


def fit_sign1(
    cc_instance: Any, dataset_code: str, sign0: Any, **fit_options: Any
) -> tuple[Any, Any]:
    """
    Clear and fit a dataset's sign1, plus the neig1 that sign2 is built from.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        CC dataset code.
    sign0 : chemicalchecker signature object
        The dataset's fitted sign0.
    **fit_options
        Forwarded to ``sign1.fit``, e.g. ``scale_kwargs`` or ``pca_kwargs``.

    Returns
    -------
    tuple of (sign1, neig1)
        The fitted sign1 and its nearest-neighbour signature.
    """
    sign1 = cc_instance.signature(dataset_code, "sign1")
    sign1.clear_all()
    sign1.fit(sign0, **fit_options)

    neig1 = cc_instance.get_signature(
        "neig1", "full", dataset_code
    )  # fits on the reference molset
    neig1.clear_all()
    neig1.fit(sign1)
    return sign1, neig1


def fit_sign2(
    cc_instance: Any,
    dataset_code: str,
    sign1: Any,
    neig1: Any,
    oos_predictor: bool = False,
    **fit_options: Any,
) -> Any:
    """
    Clear and fit a dataset's sign2 from its sign1 and neig1.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        CC dataset code.
    sign1, neig1 : chemicalchecker signature objects
        The dataset's fitted sign1 and neig1.
    oos_predictor : bool, default False
        Forwarded to ``sign2.fit``; False as in the paper's notebooks.
    **fit_options
        Other arguments forwarded to ``sign2.fit``.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign2.
    """
    sign2 = cc_instance.signature(dataset_code, "sign2")
    sign2.clear_all()
    sign2.fit(sign1, neig1, oos_predictor=oos_predictor, **fit_options)
    return sign2


def build_reference_sign2_spaces(
    cc_instance: Any, dataset_code: str, extends: str | None = None
) -> list[Any]:
    """
    List the sign2 spaces that train the dataset's sign3.

    Starts from the 25 exemplary spaces (``ChemicalChecker.datasets_exemplary``).
    A new space is appended (paper Tasks 3-4: 26 spaces); a dataset that
    extends an existing space replaces it (Tasks 1-2: 25 spaces).

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        The dataset being signaturized.
    extends : str, optional
        Code of the exemplary space this dataset extends, e.g. ``"B1.001"``.

    Returns
    -------
    list
        sign2 'full' signature objects, in exemplary order.

    Raises
    ------
    ValueError
        If ``extends`` is not an exemplary space.
    FileNotFoundError
        If any of these sign2 spaces is not available in the CC instance
        (e.g. ``custom_data_path`` was not linked).
    """
    exemplary_codes = list(cc_instance.datasets_exemplary())
    if extends is None:
        reference_codes = [*exemplary_codes, dataset_code]
    elif extends in exemplary_codes:
        reference_codes = [
            dataset_code if code == extends else code for code in exemplary_codes
        ]
    else:
        raise ValueError(
            f"{extends} is not an exemplary CC space; choose one of {exemplary_codes}"
        )
    reference_spaces = [
        cc_instance.get_signature("sign2", "full", code) for code in reference_codes
    ]
    missing = [
        code
        for code, signature in zip(reference_codes, reference_spaces)
        if not signature.available()
    ]
    if missing:
        raise FileNotFoundError(
            f"sign2 missing for {missing} in this CC instance; sign3 needs all "
            f"{len(reference_codes)} reference spaces (link them with custom_data_path)"
        )
    return reference_spaces


def fit_sign3(
    cc_instance: Any,
    dataset_code: str,
    sign2: Any,
    sign1: Any,
    reference_sign2_spaces: list[Any],
    mapping_dict: dict[str, str] | None = None,
    complete_universe: str | bool = "fast",
    **fit_options: Any,
) -> Any:
    """
    Clear and fit a dataset's sign3 (Siamese network over the reference sign2 spaces).

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        CC dataset code.
    sign2, sign1 : chemicalchecker signature objects
        The dataset's fitted sign2 and sign1 (sign1 defines the triplets).
    reference_sign2_spaces : list
        From :func:`build_reference_sign2_spaces`.
    mapping_dict : dict, optional
        InChIKey -> InChI mapping, so chemicalchecker doesn't query online repositories.
    complete_universe : {"fast", "full", False}, default "fast"
        Forwarded to ``sign3.fit``; "fast" as in the paper (skips A2 3D conformers).
    **fit_options
        Other arguments forwarded to ``sign3.fit``, e.g. ``triplets_sampler``.

    Returns
    -------
    chemicalchecker signature object
        The fitted sign3.

    Notes
    -----
    Computationally demanding (hours); normally run as a cluster job.
    """
    sign3 = cc_instance.signature(dataset_code, "sign3")
    logger.info(
        "[%s] number of sign2 spaces training sign3: %d",
        dataset_code,
        len(reference_sign2_spaces),
    )
    sign3.clear_all()
    sign3.fit(
        reference_sign2_spaces,
        sign2,
        sign1,
        complete_universe=complete_universe,
        dbconnect=False,
        mapping_dict=mapping_dict,
        **fit_options,
    )
    return sign3


# Signatures each stage is fitted from; loaded from disk when that stage is the first one fitted.
STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    "sign0": (),
    "sign1": ("sign0",),
    "sign2": ("sign1", "neig1"),
    "sign3": ("sign1", "sign2"),
}


def load_fitted_signature(cc_instance: Any, dataset_code: str, cctype: str) -> Any:
    """
    Return an already fitted signature of a dataset (the 'full' molset).

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_code : str
        CC dataset code.
    cctype : str
        Signature type, e.g. ``"sign2"`` or ``"neig1"``.

    Returns
    -------
    chemicalchecker signature object

    Raises
    ------
    FileNotFoundError
        If the signature has not been fitted in this CC instance.
    """
    signature = cc_instance.get_signature(cctype, "full", dataset_code)
    if not signature.available():
        raise FileNotFoundError(
            f"{dataset_code} {cctype} is not fitted in this CC instance "
            f"({signature.data_path}); start from an earlier stage"
        )
    logger.info("[%s] loaded fitted %s", dataset_code, cctype)
    return signature


def run_signature_pipeline(
    cc_instance: Any,
    dataset_config: DatasetConfig,
    mapping_dict: dict[str, str] | None = None,
    cc_universe: set[str] | None = None,
    diagnosis_plots: bool = True,
    start_stage: PipelineStage = "sign0",
    max_stage: PipelineStage = "sign3",
) -> dict[str, Any]:
    """
    Fit sign0 -> sign1 (+neig1) -> sign2 -> sign3 for one dataset, from ``start_stage`` to ``max_stage``.

    Parameters
    ----------
    cc_instance : chemicalchecker.ChemicalChecker
        The local CC instance.
    dataset_config : DatasetConfig
        The dataset, its data source and per-stage fit options.
    mapping_dict : dict, optional
        Forwarded to :func:`fit_sign3`.
    cc_universe : set of str, optional
        If given, the dataset's overlap with it is logged after sign2.
    diagnosis_plots : bool, default True
        Save CC diagnosis canvases after each fitted stage.
    start_stage : {"sign0", "sign1", "sign2", "sign3"}, default "sign0"
        First signature type to fit. The signatures it is fitted from (see
        :data:`STAGE_INPUTS`) are loaded from the CC instance instead of refitted,
        e.g. "sign3" fits sign3 on the existing sign1 and sign2.
    max_stage : {"sign0", "sign1", "sign2", "sign3"}, default "sign3"
        Last signature type to fit (e.g. "sign2" to skip the costly sign3).

    Returns
    -------
    dict
        Signature objects keyed by type (``"sign0"``, ``"sign1"``, ``"neig1"``,
        ``"sign2"``, ``"sign3"``): the loaded inputs of ``start_stage`` followed
        by the fitted stages.

    Raises
    ------
    ValueError
        If the stages are not pipeline stages or ``start_stage`` comes after ``max_stage``.
    FileNotFoundError
        If a signature needed by ``start_stage`` has not been fitted yet.
    """
    check_stage_range(start_stage, max_stage)
    dataset_code, fit_options = dataset_config.dataset_code, dataset_config.fit
    label = f"{dataset_config.name} ({dataset_code})"
    stages = PIPELINE_STAGES[
        PIPELINE_STAGES.index(start_stage) : PIPELINE_STAGES.index(max_stage) + 1
    ]
    signatures: dict[str, Any] = {
        cctype: load_fitted_signature(cc_instance, dataset_code, cctype)
        for cctype in STAGE_INPUTS[start_stage]
    }

    for stage in stages:
        if stage == "sign0":
            signatures["sign0"] = fit_sign0(
                cc_instance,
                dataset_code,
                build_sign0_inputs(dataset_config.source),
                **fit_options.sign0,
            )
        elif stage == "sign1":
            signatures["sign1"], signatures["neig1"] = fit_sign1(
                cc_instance, dataset_code, signatures["sign0"], **fit_options.sign1
            )
        elif stage == "sign2":
            signatures["sign2"] = fit_sign2(
                cc_instance,
                dataset_code,
                signatures["sign1"],
                signatures["neig1"],
                **fit_options.sign2,
            )
            if cc_universe is not None:
                report_universe_overlap(label, signatures["sign2"], cc_universe)
        else:
            extends = (
                dataset_config.reference_spaces.extends
                if isinstance(dataset_config.reference_spaces, ExtendedSpace)
                else None
            )
            signatures["sign3"] = fit_sign3(
                cc_instance,
                dataset_code,
                signatures["sign2"],
                signatures["sign1"],
                build_reference_sign2_spaces(
                    cc_instance, dataset_code, extends=extends
                ),
                mapping_dict=mapping_dict,
                **fit_options.sign3,
            )
        report_minmax(signatures[stage], label=f"{label} {stage}")
        if diagnosis_plots:
            if stage == "sign3":
                diagnose_and_plot(
                    signatures["sign3"], sizes=("medium", "small"), ref_cctype="sign3"
                )
            else:
                diagnose_and_plot(signatures[stage])
    return signatures
