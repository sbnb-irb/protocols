"""Logging setup for pipeline runs, including restoring logs after importing chemicalchecker."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


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
    now: datetime = datetime.now()  # noqa: DTZ005 -- local wall-clock time for file names
    timezone: str = now.astimezone().tzname() or ""
    stamp = (
        now.isoformat(sep="_", timespec="seconds")
        if incl_time
        else now.date().isoformat()
    )
    if incl_timezone and timezone:
        stamp = f"{stamp}_{timezone}"
    return stamp.replace(":", "-")  # ':' is invalid in filenames on some systems


def generate_log_filename(
    folder: str | os.PathLike[str] = "logs", suffix: str = ""
) -> Path:
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


def setup_logging(
    log_file_path: str | os.PathLike[str], display: bool = True, mode: str = "w"
) -> logging.Logger:
    """
    Configures the root logger with a file handler and optional console handler.

    Parameters
    ----------
    log_file_path : str or os.PathLike
        Path to the log file to write.
    display : bool, default True
        Whether to also stream log records to the console (stdout).
    mode : str, default "w"
        Mode for the log file handler. Use "a" (append) when re-establishing
        logging after something already wrote to this same path this run --
        see :func:`resume_logging_after_import`, whose whole point is to
        reopen the same file without truncating what's already in it.

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
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    ]
    if display:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[
        :
    ]:  # reset so repeated runs don't stack handlers
        root_logger.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.info("Path to log file: %s", log_path)
    return root_logger


def resume_logging_after_import(
    log_file_path: str | os.PathLike[str], display: bool = True
) -> logging.Logger:
    """
    Restore this project's logging after importing/using ``chemicalchecker``.

    Importing ``chemicalchecker`` runs a module-level
    ``logging.config.fileConfig(...)`` call (see
    ``chemicalchecker.util.logging.our_logging``), which uses the stdlib
    default ``disable_existing_loggers=True``. That silently (a) sets
    ``.disabled = True`` on every logger object that already existed at that
    point -- including this package's module-level loggers, created via
    ``logging.getLogger(__name__)`` before the chemicalchecker import runs --
    and (b) closes and replaces the root logger's own handlers outright with
    chemicalchecker's single stderr handler. Without this, every
    ``logger.info()`` call in this package silently vanishes for the rest of
    the process, even though chemicalchecker's own logging keeps working.

    Call this once, right after importing (or otherwise first invoking
    ``ChemicalChecker.set_verbosity``/etc.) chemicalchecker, passing the
    exact same path given to the earlier :func:`setup_logging` call. Reopens
    the log file in append mode -- the original handler's file stream is
    already closed by that point, so it has to be a new handler, not the
    same object -- so nothing already written is lost.

    Parameters
    ----------
    log_file_path : str or os.PathLike
        The same path passed to the earlier :func:`setup_logging` call.
    display : bool, default True
        Forwarded to :func:`setup_logging`.

    Returns
    -------
    logging.Logger
        The re-configured root logger.
    """
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.disabled = False
    return setup_logging(log_file_path, display=display, mode="a")


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
