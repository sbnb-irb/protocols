"""Loaders that read a dataset's raw bioactivity data and hand it to ``sign0.fit``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .config import DataSource

logger = logging.getLogger(__name__)


def read_table_with_required_columns(
    input_path: str | os.PathLike[str], columns: list[str], **read_csv_kwargs: Any
) -> pd.DataFrame:
    """
    Reads a delimited text file, asserting it exists and has the required columns.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input CSV/TSV.
    columns : list of str
        Columns that must be present in the file (after ``index_col`` is applied).
    **read_csv_kwargs
        Extra keyword arguments forwarded to ``pandas.read_csv`` (separator,
        index column, encoding...).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist or is not a file.
    KeyError
        If any required column is missing.
    """
    logger.info("Reading input table from: %s", input_path)
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Could not find input file: {input_path}")
    dataframe = pd.read_csv(input_path, **read_csv_kwargs)
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise KeyError(
            f"Columns {missing} not found in {Path(input_path).name}. "
            f"First columns available: {list(dataframe.columns[:12])}"
        )
    logger.info("Loaded %d rows x %d columns", dataframe.shape[0], dataframe.shape[1])
    return dataframe


def load_wide_matrix(source: DataSource) -> pd.DataFrame:
    """
    Load a compound x feature matrix with InChIKeys in the first column.

    Parameters
    ----------
    source : DataSource
        A ``wide_matrix`` source.

    Returns
    -------
    pd.DataFrame
        Rows indexed by InChIKey, one column per feature.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the matrix has no rows or no feature columns.
    """
    feature_matrix = read_table_with_required_columns(
        source.path, columns=[], sep=source.separator, index_col=0
    )
    if feature_matrix.empty:
        raise ValueError(
            f"{source.path} has {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} "
            "feature columns; expected compounds as rows and features as columns"
        )
    return feature_matrix


def build_sign0_inputs(source: DataSource) -> dict[str, Any]:
    """
    Translate a data source into the keyword arguments ``sign0.fit`` expects.

    Wide matrices are read here (with the checks of :func:`load_wide_matrix`);
    ``long_pairs`` and ``cc_h5`` files are passed as paths, because
    chemicalchecker reads those formats itself. Their existence is checked
    first, since chemicalchecker's own error for a missing file is unclear.

    Parameters
    ----------
    source : DataSource
        The dataset's raw data.

    Returns
    -------
    dict
        ``{"X", "keys", "features"}``, ``{"pairs"}`` or ``{"data_file"}``.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    """
    if source.format == "wide_matrix":
        feature_matrix = load_wide_matrix(source)
        return {
            "X": feature_matrix.values,
            "keys": list(feature_matrix.index),
            "features": list(feature_matrix.columns),
        }
    if not source.path.is_file():
        raise FileNotFoundError(
            f"Could not find {source.format} input file: {source.path}"
        )
    logger.info("Passing %s input %s to chemicalchecker", source.format, source.path)
    argument = "pairs" if source.format == "long_pairs" else "data_file"
    return {argument: str(source.path)}


def load_inchikey_mapping(
    mapping_path: str | os.PathLike[str] | None,
) -> dict[str, str] | None:
    """
    Load an optional InChIKey -> InChI mapping from a JSON file.

    Parameters
    ----------
    mapping_path : str or os.PathLike or None
        The JSON file, or None to skip.

    Returns
    -------
    dict or None
        The mapping, or None if ``mapping_path`` is None.

    Raises
    ------
    FileNotFoundError
        If ``mapping_path`` is given but is not a file.
    """
    if mapping_path is None:
        return None
    if not Path(mapping_path).is_file():
        raise FileNotFoundError(f"InChIKey mapping JSON not found at: {mapping_path}")
    with open(mapping_path, encoding="utf-8") as handle:
        inchikey_mapping = json.load(handle)
    logger.info(
        "Loaded InChIKey->InChI mapping with %d entries from %s",
        len(inchikey_mapping),
        mapping_path,
    )
    return inchikey_mapping
