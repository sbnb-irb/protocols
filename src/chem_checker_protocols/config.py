"""Run and dataset configuration for the signature-fitting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import pandas as pd
import yaml

from .data_loaders import check_path

logger = logging.getLogger(__name__)


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


DEFAULT_DATASET_CONFIG: Path = Path(__file__).resolve().parent / "pertprot_datasets.yaml"


_REQUIRED_REGISTRY_FIELDS = {"key", "name", "code", "csv_path"}


def load_dataset_registry(
    config_path: str | os.PathLike[str] = DEFAULT_DATASET_CONFIG,
) -> list[dict[str, str]]:
    """
    Load and validate the PerturbProt dataset registry from a YAML file.

    Parameters
    ----------
    config_path : str or os.PathLike, default :data:`DEFAULT_DATASET_CONFIG`
        Path to a YAML file with a top-level ``datasets:`` list, each entry
        providing ``key``, ``name``, ``code`` and ``csv_path`` (see
        ``pertprot_datasets.yaml`` for the schema and field meanings).

    Returns
    -------
    list of dict
        The raw ``datasets`` entries, in file order.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the file has no (or an empty) ``datasets`` list.
    KeyError
        If any entry is missing a required field.
    """
    config_path = Path(config_path)
    if not check_path(config_path):
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as fh:
        registry = yaml.safe_load(fh)

    entries = registry.get("datasets") if isinstance(registry, dict) else None
    if not entries:
        raise ValueError(f"Dataset config {config_path} has no 'datasets' entries.")
    for entry in entries:
        missing = _REQUIRED_REGISTRY_FIELDS - entry.keys()
        if missing:
            raise KeyError(f"Dataset entry {entry} in {config_path} missing required fields: {missing}")
    logger.info("Loaded %d dataset entries from %s", len(entries), config_path)
    return entries


def dataset_keys(config_path: str | os.PathLike[str] = DEFAULT_DATASET_CONFIG) -> tuple[str, ...]:
    """
    Return the valid dataset keys defined in a registry, in file order.

    Parameters
    ----------
    config_path : str or os.PathLike, default :data:`DEFAULT_DATASET_CONFIG`
        Forwarded to :func:`load_dataset_registry`.

    Returns
    -------
    tuple of str
        Every ``key`` field in the registry.
    """
    return tuple(entry["key"] for entry in load_dataset_registry(config_path))


def default_dataset_specs(
    data_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = DEFAULT_DATASET_CONFIG,
) -> list[DatasetSpec]:
    """
    Build PerturbProt DatasetSpecs for every entry in a dataset registry.

    Parameters
    ----------
    data_dir : str or os.PathLike
        Base directory containing the wide compound-matrix CSVs (as produced
        by ``transform_allstudies_to_widedf.py``); each entry's ``csv_path``
        is resolved relative to this directory.
    config_path : str or os.PathLike, default :data:`DEFAULT_DATASET_CONFIG`
        YAML dataset registry, forwarded to :func:`load_dataset_registry`.

    Returns
    -------
    list of DatasetSpec
        One spec per registry entry, in file order.
    """
    data_dir = Path(data_dir)
    entries = load_dataset_registry(config_path)
    return [
        DatasetSpec(key=entry["key"], name=entry["name"], code=entry["code"], csv_path=data_dir / entry["csv_path"])
        for entry in entries
    ]
