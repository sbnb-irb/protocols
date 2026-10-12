"""Read the Chemical Checker's own diagnostics of a fitted signature from its files.

``sign.diagnosis()`` writes pickles under ``<signature>/diags/<name>_<cctype>/`` and
``sign.validate()`` writes ``<signature>/stats/validation_stats.json``. These are CC's metrics,
computed on the whole universe, not on held-out molecules:

- neighbourhood AUROC (``moa_roc``, ``atc_roc``, ``across_roc``): each molecule's nearest
  neighbours in another CC space's signature against random pairs, on CC's subsample;
- annotation AUROC (``validation_stats.json``): curated MoA/ATC molecule pairs;
- per-molecule confidence (``confidences``).

Reading the files needs no chemicalchecker import, so it also works outside the CC container.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def signature_folder(
    cc_root: str | os.PathLike[str],
    dataset_code: str,
    cctype: str = "sign3",
    molset: str = "full",
) -> Path:
    """
    Folder of one signature in a local CC instance, e.g. ``<cc_root>/full/D/D6/D6.002/sign3``.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    folder = (
        Path(cc_root)
        / molset
        / dataset_code[0]
        / dataset_code[:2]
        / dataset_code
        / cctype
    )
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No {cctype} of {dataset_code} in {cc_root} (looked for {folder})"
        )
    return folder


def read_cc_diagnostics(
    cc_root: str | os.PathLike[str], dataset_code: str, cctype: str = "sign3"
) -> pd.DataFrame:
    """
    Collect CC's validation numbers of one signature into a long table.

    Parameters
    ----------
    cc_root : str or os.PathLike
        Local CC instance.
    dataset_code : str
        CC dataset code, e.g. ``"D6.002"``.
    cctype : str, default "sign3"
        Signature type.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``cc_space`` and ``value``. Metrics: ``neighbourhood_auroc``
        (``cc_space`` = B1.001 for MoA, E1.001 for ATC), ``across_space_auroc`` (one row per
        CC space), ``annotation_auroc_moa`` / ``annotation_auroc_atc`` and
        ``mean_confidence``; files that are missing are skipped with a warning.
    """
    folder = signature_folder(cc_root, dataset_code, cctype)
    diags = sorted((folder / "diags").glob(f"*_{cctype}"))
    rows = []
    if not diags:
        logger.warning("No diagnosis folder in %s/diags", folder)
    else:
        diag = diags[0]
        for name, cc_space in (("moa_roc", "B1.001"), ("atc_roc", "E1.001")):
            result = _read_pickle(diag / f"{name}.pkl")
            if result is not None:
                rows.append(("neighbourhood_auroc", cc_space, float(result["auc"])))
        across = _read_pickle(diag / "across_roc.pkl")
        if across is not None:
            rows += [
                ("across_space_auroc", space, float(result["auc"]))
                for space, result in across.items()
                if result is not None
            ]
        confidences = _read_pickle(diag / "confidences.pkl")
        if confidences is not None:
            rows.append(
                (
                    "mean_confidence",
                    "",
                    float(pd.Series(confidences["confidences"]).mean()),
                )
            )
    stats_file = folder / "stats" / "validation_stats.json"
    if stats_file.is_file():
        stats = json.loads(stats_file.read_text())
        rows += [
            (f"annotation_auroc_{kind}", "", float(stats[f"{kind}_auc"]))
            for kind in ("moa", "atc")
            if f"{kind}_auc" in stats
        ]
    else:
        logger.warning("No %s", stats_file)
    return pd.DataFrame(rows, columns=["metric", "cc_space", "value"])


def _read_pickle(path: Path):
    if not path.is_file():
        logger.warning("No %s", path)
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)
