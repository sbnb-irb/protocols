#!/usr/bin/env python3
"""Compares two Chemical Checker bioactivity spaces (e.g. a space rebuilt
with additional data against its earlier version) using the diagnosis
artifacts already written by ``signature.diagnosis().canvas()`` plus a
shared-compound nearest-neighbor recapitulation test (Comajuncosa-Creus et
al., Nat. Protoc. 2025, Fig. 3 / Extended Data Fig. 8).

All comparison logic lives in ``cc_compare_claudecode.py``; this script only wires
together CLI configuration and calls ``run_comparison()``. It is meant to be
run after both spaces have already been fitted and diagnosed (e.g. via
``pertprot_cli.py`` or an equivalent pipeline script) -- it does not refit
any signatures, it only reads existing sign3 signatures and diagnosis
artifacts from disk.

Example
-------
python compare_cc_spaces_claudecode.py \\
    --local-cc-dir /scratch/sbnb/sayala/protocols/local_CC_D6 \\
    --dataset-a D6.002 --label-a "DeepCoverMoA only" \\
    --dataset-b D6.007 --label-b "Extended proteomics" \\
    --output-dir results/D6.002_vs_D6.007
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

# Make cc_compare_claudecode.py (in this same directory) importable, and also add the
# parent directory so utils.py -- shared with pertprot_cli.py/cc_pipeline.py
# and not duplicated into this subfolder -- stays importable even though
# this script now lives one level down (e.g. scripts/cc_space_comparison/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from cc_compare_claudecode import run_comparison
from utils import generate_log_filename, get_basename, log_run, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the space-vs-space comparison.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line argument list for programmatic testing. Defaults to
        ``sys.argv[1:]`` when None.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare two Chemical Checker bioactivity spaces using their "
        "existing diagnosis artifacts and a shared-compound NN recapitulation test."
    )
    parser.add_argument(
        "--cc-config",
        type=Path,
        default=Path("/scratch/sbnb/sayala/chemical_checker/setup/cc_config.json"),
        help="Path to the Chemical Checker cc_config.json file.",
    )
    parser.add_argument(
        "--local-cc-dir",
        type=Path,
        default=None,
        help="Local Chemical Checker instance directory shared by both spaces. "
        "Use this when both dataset codes live in the same instance "
        "(e.g. D6.002 and D6.007 both under local_CC_D6). Ignored if "
        "--local-cc-dir-a and --local-cc-dir-b are both given instead.",
    )
    parser.add_argument(
        "--local-cc-dir-a",
        type=Path,
        default=None,
        help="Local CC instance directory for space A, if different from --local-cc-dir.",
    )
    parser.add_argument(
        "--local-cc-dir-b",
        type=Path,
        default=None,
        help="Local CC instance directory for space B, if different from --local-cc-dir.",
    )
    parser.add_argument(
        "--dataset-a", type=str, required=True, help="CC dataset code for space A (e.g. D6.002)."
    )
    parser.add_argument(
        "--dataset-b", type=str, required=True, help="CC dataset code for space B (e.g. D6.007)."
    )
    parser.add_argument(
        "--label-a", type=str, default=None, help="Display label for space A (default: dataset code)."
    )
    parser.add_argument(
        "--label-b", type=str, default=None, help="Display label for space B (default: dataset code)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write comparison tables/figures to (created if missing).",
    )
    parser.add_argument(
        "--pval",
        type=float,
        default=0.01,
        help="P-value of the background pairwise cosine-distance distribution used "
        "as the nearest-neighbor cutoff in the shared-key recapitulation test "
        "(default: 0.01, as in the CC Protocols paper). The cutoff is estimated "
        "once from ~10000 random compound pairs x 10 subsamples and then applied "
        "as a fixed absolute distance threshold.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=2500,
        help="Number of randomly sampled shared compounds per recapitulation subsample.",
    )
    parser.add_argument(
        "--n-subsamples",
        type=int,
        default=5,
        help="Number of subsamples to average the recapitulation AUROC over.",
    )
    parser.add_argument(
        "--max-pool",
        type=int,
        default=50000,
        help="Cap on how many shared compounds are loaded into memory for the "
        "recapitulation test (default: 50000). Two sign3 spaces each cover the "
        "whole ~1.2M-compound CC universe, so loading every shared key for both "
        "costs ~1.2 GB of RAM; the test itself only ever samples --n-random "
        "compounds at a time, so a random pool of this size is statistically "
        "equivalent and far cheaper. Pass 0 to load all shared compounds.",
    )
    parser.add_argument(
        "--random-state", type=int, default=None, help="Random seed for reproducible subsampling."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing comparison figures; the CSV/JSON tables are always written.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="DEBUG",
        # These must match ChemicalChecker.set_verbosity()'s own level map
        # exactly -- it does a bare levels[level] lookup, so an almost-right
        # value like "WARN" raises KeyError instead of degrading gracefully.
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Chemical Checker verbosity level (default: DEBUG).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store execution log files (default: logs).",
    )
    return parser.parse_args(argv)


def _resolve_cc_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Resolve per-space local CC instance directories from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    tuple of (Path, Path)
        Local CC instance directories for space A and space B.

    Raises
    ------
    ValueError
        If neither ``--local-cc-dir`` nor both of ``--local-cc-dir-a`` /
        ``--local-cc-dir-b`` are given.
    """
    if args.local_cc_dir_a is not None and args.local_cc_dir_b is not None:
        return args.local_cc_dir_a, args.local_cc_dir_b
    if args.local_cc_dir is not None:
        return args.local_cc_dir, args.local_cc_dir
    raise ValueError(
        "Provide either --local-cc-dir (both spaces in one instance) or "
        "both --local-cc-dir-a and --local-cc-dir-b."
    )


def _restore_root_logging(
    log_path: Path,
    level: int,
    reenable: tuple[str, ...] = ("__main__", "utils", "cc_compare_claudecode"),
) -> None:
    """Undo the logging teardown caused by importing ``chemicalchecker``.

    ``chemicalchecker/util/logging/our_logging.py`` calls
    ``logging.config.fileConfig()`` at import time. That one call does two
    destructive things, and both have to be undone or the log file ends up
    essentially empty -- which is exactly what every run of this script
    produced before this function existed:

    1. **It replaces the root logger's handlers** with a single stderr
       handler, discarding the file handler ``utils.setup_logging()``
       installed, and *closes* the handlers it removes. Closed handlers
       cannot simply be re-attached -- records route to them and vanish
       without error -- so the handlers here are rebuilt from scratch. The
       file handler opens in append mode to keep the lines written before
       the import.
    2. **It disables every logger that already exists**, because
       ``fileConfig`` defaults to ``disable_existing_loggers=True``. Any
       module-level ``logging.getLogger(__name__)`` evaluated before the
       import -- which is all of ours -- comes back with ``disabled = True``
       and silently drops every record. This is the subtler half: the root
       logger looks perfectly healthy afterwards (right handlers, right
       level, ``manager.disable == 0``), so the cause is invisible unless
       the individual logger's ``disabled`` flag is checked.

    Only this project's own loggers are re-enabled, not every disabled
    logger, so third-party libraries silenced by that config stay silent.

    Parameters
    ----------
    log_path : Path
        Log file to resume writing to, in append mode.
    level : int
        Root logger level captured before the import.
    reenable : tuple of str, default ("__main__", "utils", "cc_compare_claudecode")
        Logger names to clear the ``disabled`` flag on. ``"__main__"``
        covers this script when run directly; the others are the shared
        helper module and the comparison logic module.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler, stream_handler]
    root_logger.setLevel(level)

    for name in reenable:
        logging.getLogger(name).disabled = False

    # chemicalchecker keeps its own handler; without this its records would
    # be emitted twice, once by it and once by the rebuilt root handlers.
    logging.getLogger("chemicalchecker").propagate = False


def main(argv: Sequence[str] | None = None) -> dict:
    """
    Run the space-vs-space comparison end-to-end.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments for programmatic testing.

    Returns
    -------
    dict
        The dictionary returned by :func:`cc_compare_claudecode.run_comparison`
        (summary table, across-CC comparison, recapitulation results).
    """
    args = parse_args(argv)

    log_path = Path(generate_log_filename(args.log_dir, suffix=get_basename(__file__)))
    setup_logging(log_path)

    with log_run():
        local_cc_dir_a, local_cc_dir_b = _resolve_cc_dirs(args)
        # NOTE: deliberately not using utils.check_path() here -- that helper
        # is written for file paths (see its use for --mapping-json in
        # pertprot_cli.py) and returns False for directories even when they
        # exist. A local CC instance directory is a directory, so check it
        # directly instead.
        for path in {local_cc_dir_a, local_cc_dir_b}:
            if not path.is_dir():
                raise FileNotFoundError(f"Local CC instance directory not found: {path}")

        # CC_CONFIG must be set before chemicalchecker is imported
        os.environ["CC_CONFIG"] = str(args.cc_config)
        # The import below tears down (and closes) the root logger's
        # handlers -- see _restore_root_logging.
        saved_level = logging.getLogger().level
        from chemicalchecker import ChemicalChecker

        ChemicalChecker.set_verbosity(args.verbosity)
        _restore_root_logging(log_path, saved_level)

        label_a = args.label_a or args.dataset_a
        label_b = args.label_b or args.dataset_b

        logger.info(
            "Comparing %s (%s, %s) vs %s (%s, %s)",
            label_a,
            args.dataset_a,
            local_cc_dir_a,
            label_b,
            args.dataset_b,
            local_cc_dir_b,
        )

        results = run_comparison(
            local_cc_dir_a=local_cc_dir_a,
            dataset_a=args.dataset_a,
            local_cc_dir_b=local_cc_dir_b,
            dataset_b=args.dataset_b,
            label_a=label_a,
            label_b=label_b,
            output_dir=args.output_dir,
            pval=args.pval,
            n_random=args.n_random,
            n_subsamples=args.n_subsamples,
            random_state=args.random_state,
            max_pool=args.max_pool,
            make_plots=not args.no_plots,
        )

        logger.info("Comparison complete. Outputs written to %s", args.output_dir)

    return results


if __name__ == "__main__":
    main()