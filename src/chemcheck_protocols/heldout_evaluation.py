"""Held-out evaluation of several runs against several references, from an ``EvaluationConfig``.

The scoring itself is in :mod:`chemcheck_protocols.evaluation`; this module loads the vectors:
which molecules take part, the reference vectors (a CC signature or binarised profiles), each
run's sign3 (one instance or fold models), and writes the result tables.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cc_diagnostics import read_cc_diagnostics
from .config import EvaluationConfig, EvaluationReference
from .data_loaders import load_key_list, load_wide_matrix
from .evaluation import (
    paired_difference,
    per_molecule_heldout_auroc,
    pooled_heldout_auroc,
)
from .triplet_profiles import binarize_profiles

logger = logging.getLogger(__name__)

SignatureOpener = Callable[[Path, str, str], Any]
"""``open_signature(cc_root, dataset_code, cctype)`` -> object with ``keys`` and
``get_vectors(keys) -> (keys, vectors)``, like a chemicalchecker signature."""


def cc_signature_opener() -> SignatureOpener:
    """Open signatures with chemicalchecker, one ``ChemicalChecker`` per instance."""
    from chemicalchecker import ChemicalChecker

    instances: dict[str, Any] = {}

    def open_signature(cc_root: Path, dataset_code: str, cctype: str) -> Any:
        root = str(cc_root)
        if root not in instances:
            instances[root] = ChemicalChecker(root, dbconnect=False)
        return instances[root].get_signature(cctype, "full", dataset_code)

    return open_signature


def run_evaluation(
    config: EvaluationConfig, open_signature: SignatureOpener
) -> dict[str, pd.DataFrame]:
    """
    Score every run on every reference.

    Parameters
    ----------
    config : EvaluationConfig
        The validated evaluation.
    open_signature : SignatureOpener
        How signatures are opened (``cc_signature_opener()`` in production).

    Returns
    -------
    dict of str to pandas.DataFrame
        ``scores`` (one row per reference and run), ``comparisons`` (paired differences:
        every run against the baseline, then the configured pairs) and, if requested,
        ``per_molecule`` and ``cc_diagnostics``.
    """
    space_keys = set(
        open_signature(config.reference_cc, config.dataset_code, "sign0").keys
    )
    sign3s = {}
    for run in config.runs:
        if run.folds:
            sign3s[run.name] = [
                open_signature(
                    f.cc_root, run.dataset_code or config.dataset_code, "sign3"
                )
                for f in run.folds
            ]
        else:
            sign3s[run.name] = open_signature(
                run.cc_root, run.dataset_code or config.dataset_code, "sign3"
            )
    all_sign3 = [
        s
        for value in sign3s.values()
        for s in (value if isinstance(value, list) else [value])
    ]
    predicted = set.intersection(*(set(s.keys) for s in all_sign3))
    fold_run = next((run for run in config.runs if run.folds), None)
    if fold_run is not None:
        hidden = [load_key_list(fold.holdout) for fold in fold_run.folds]
        logger.info(
            "Held-out molecules from %d folds: %s",
            len(hidden),
            [len(h) for h in hidden],
        )
    else:
        hidden = [load_key_list(config.holdout)]

    scores, comparisons, per_molecule = [], [], []
    for reference in config.references:
        ref_keys, ref_lookup = _reference(config, reference, open_signature)
        keys = np.array(sorted(space_keys & predicted & ref_keys))
        ref_vectors = ref_lookup(keys)
        query = {
            name: (
                [s.get_vectors(keys)[1] for s in value]
                if isinstance(value, list)
                else value.get_vectors(keys)[1]
            )
            for name, value in sign3s.items()
        }
        defined = np.linalg.norm(ref_vectors, axis=1) > 0
        for value in query.values():
            for matrix in value if isinstance(value, list) else [value]:
                defined &= np.linalg.norm(matrix, axis=1) > 0
        if not defined.all():
            logger.warning(
                "%s: leaving out %d molecules with an all-zero vector",
                reference.name,
                int((~defined).sum()),
            )
        keys, ref_vectors = keys[defined], ref_vectors[defined]
        query = {
            name: (
                [m[defined] for m in value]
                if isinstance(value, list)
                else value[defined]
            )
            for name, value in query.items()
        }
        fold_rows = [np.flatnonzero(np.isin(keys, fold_keys)) for fold_keys in hidden]
        heldout_rows = np.concatenate(fold_rows)
        fold_of_row = np.concatenate(
            [np.full(len(rows), i) for i, rows in enumerate(fold_rows)]
        )
        run_vectors = {
            name: (
                [value[i] for i in fold_of_row] if isinstance(value, list) else value
            )
            for name, value in query.items()
        }
        logger.info(
            "%s: %d held-out molecules among %d",
            reference.name,
            len(heldout_rows),
            len(keys),
        )
        results = pooled_heldout_auroc(
            ref_vectors,
            heldout_rows,
            run_vectors,
            p_value_cutoff=config.p_value_cutoff,
            n_bootstrap=config.n_bootstrap,
            random_state=config.seed,
        )
        for name, result in results.items():
            scores.append(
                {
                    "reference": reference.name,
                    "run": name,
                    **{k: v for k, v in result.items() if k != "bootstrap_aurocs"},
                }
            )
        pairs = [
            (name, config.baseline_run)
            for name in results
            if name != config.baseline_run
        ]
        for first, second in pairs + [tuple(pair) for pair in config.comparisons]:
            comparisons.append(
                {
                    "reference": reference.name,
                    "first": first,
                    "second": second,
                    **paired_difference(results[first], results[second]),
                }
            )
        if config.per_molecule:
            single = {
                name: value
                for name, value in query.items()
                if not isinstance(value, list)
            }
            table = per_molecule_heldout_auroc(
                ref_vectors,
                heldout_rows,
                single,
                p_value_cutoff=config.p_value_cutoff,
                random_state=config.seed,
            )
            table.insert(0, "inchikey", keys[table.pop("row").to_numpy()])
            table.insert(0, "reference", reference.name)
            per_molecule.append(table)

    tables = {"scores": pd.DataFrame(scores), "comparisons": pd.DataFrame(comparisons)}
    if config.per_molecule:
        tables["per_molecule"] = pd.concat(per_molecule, ignore_index=True)
    if config.cc_diagnostics:
        tables["cc_diagnostics"] = pd.concat(
            [
                read_cc_diagnostics(
                    run.cc_root, run.dataset_code or config.dataset_code
                ).assign(run=run.name)
                for run in config.runs
                if not run.folds
            ],
            ignore_index=True,
        )
    return tables


def write_evaluation(
    tables: dict[str, pd.DataFrame], output: str | os.PathLike[str]
) -> list[Path]:
    """Write each table to ``<output>_<name>.csv`` and return the paths."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = output.parent / f"{output.name}_{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    logger.info("Wrote %s", ", ".join(str(p) for p in paths))
    return paths


def _reference(
    config: EvaluationConfig,
    reference: EvaluationReference,
    open_signature: SignatureOpener,
) -> tuple[set[str], Callable[[np.ndarray], np.ndarray]]:
    """Keys of a reference and a function returning its vectors for given keys."""
    if reference.signature is not None:
        signature = open_signature(
            reference.cc_root or config.reference_cc,
            reference.dataset_code or config.dataset_code,
            reference.signature,
        )
        return set(signature.keys), lambda keys: signature.get_vectors(keys)[1]
    profiles = load_wide_matrix(reference.profiles).groupby(level=0).mean()
    if reference.binarize is not None:
        profiles = binarize_profiles(profiles, reference.binarize)
    elif profiles.isna().to_numpy().any():
        raise ValueError(
            f"reference {reference.name!r} has missing values; add binarize or fill them"
        )
    return set(profiles.index), lambda keys: profiles.loc[keys].to_numpy(dtype=float)
