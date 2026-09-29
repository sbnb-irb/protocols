"""Loaders that read raw bioactivity data into the signature-fitting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import logging
import os
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DatasetSpec

logger = logging.getLogger(__name__)


def check_path(filepath: str | os.PathLike[str]) -> bool:
    """
    Checks if the given file or folder path exists.

    Parameters
    ----------
    filepath : str or os.PathLike
        The file or folder path to check.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    return os.path.exists(os.path.abspath(filepath))


def read_table_with_required_columns(
    input_path: str | os.PathLike[str], columns: list[str], **read_csv_kwargs
) -> pd.DataFrame:
    """
    Reads a delimited text file, asserting it exists and has the required columns.

    Delimiter, encoding and header offset are caller-supplied via
    ``read_csv_kwargs`` (see ``STUDY_REGISTRY`` in ``schema.py``), because the
    upstream supplementary files are inconsistent on all three.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input CSV/TSV.
    columns : list of str
        Columns that must be present in the file.
    **read_csv_kwargs
        Extra keyword arguments forwarded to ``pandas.read_csv``.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    KeyError
        If any required column is missing.
    """
    logger.info("Reading input table from: %s", input_path)
    if not check_path(input_path):
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
