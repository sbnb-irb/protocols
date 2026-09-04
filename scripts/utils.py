"""
Shared utilities for the PerturbProt Chemical Checker signature pipeline.

Generic, script-agnostic plumbing reused by ``cc_pipeline.py``,
``pertprot_cli.py`` and ``pertprot_interactive.py``: path checks,
timestamped per-run logging, a run-timer context manager, and CSV loading
with column validation. Trimmed from the shared ``globalprot-data-prep``
utils module (used by ``harmonize_datasets.py`` and sibling projects such as
``perturbprot-data-integration``) down to only what this pipeline uses --
notably, no Parquet I/O helpers, since this pipeline only reads wide CSVs.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def get_basename(fname: str | os.PathLike[str] | None = None) -> str:
    """
    Returns the basename of a path without its file extension.

    Parameters
    ----------
    fname : str or os.PathLike, optional
        The path to derive the basename from. If None, the basename of the
        currently executing script is used.

    Returns
    -------
    str
        The basename without extension.
    """
    target: str = str(fname) if fname is not None else sys.argv[0]
    return os.path.splitext(os.path.basename(target))[0]


def get_time(incl_time: bool = True, incl_timezone: bool = True) -> str:
    """
    Builds a filesystem-safe timestamp string for naming log files.

    Parameters
    ----------
    incl_time : bool, default True
        Whether to include the time (hh-mm-ss) in addition to the date.
    incl_timezone : bool, default True
        Whether to append the local timezone abbreviation.

    Returns
    -------
    str
        A timestamp such as ``2026-07-29_16-40-56_CEST``.
    """
    now: datetime = datetime.now()
    timezone: str = now.astimezone().tzname() or ""
    stamp = now.isoformat(sep="_", timespec="seconds") if incl_time else now.date().isoformat()
    if incl_timezone and timezone:
        stamp = f"{stamp}_{timezone}"
    return stamp.replace(":", "-")  # ':' is invalid in filenames on some systems


def generate_log_filename(folder: str | os.PathLike[str] = "logs", suffix: str = "") -> Path:
    """
    Creates a timestamped log-file path inside a folder, creating the folder.

    Parameters
    ----------
    folder : str or os.PathLike, default "logs"
        Directory to place the log file in (created if missing).
    suffix : str, default ""
        Extra identifier appended after the timestamp, e.g. the script basename.

    Returns
    -------
    pathlib.Path
        The full path to the log file.
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    stem: str = get_time(incl_timezone=False)
    if suffix:
        stem = f"{stem}_{suffix}"
    return folder_path / f"{stem}.log"


def setup_logging(log_file_path: str | os.PathLike[str], display: bool = True) -> logging.Logger:
    """
    Configures the root logger with a file handler and optional console handler.

    Parameters
    ----------
    log_file_path : str or os.PathLike
        Path to the log file to write.
    display : bool, default True
        Whether to also stream log records to the console (stdout).

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    if display:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:  # reset so repeated runs don't stack handlers
        root_logger.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.info("Path to log file: %s", log_path)
    return root_logger


@contextmanager
def log_run() -> Iterator[None]:
    """
    Context manager that logs the command line on entry and run time on exit.

    Yields
    ------
    None
    """
    logger.info("Command line: %s", " ".join(sys.argv))
    start_time: float = time.perf_counter()
    try:
        yield
    finally:
        logger.info("Total run time: %.1f s", time.perf_counter() - start_time)


# --------------------------------------------------------------------------- #
# DataFrame I/O                                                                #
# --------------------------------------------------------------------------- #
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
