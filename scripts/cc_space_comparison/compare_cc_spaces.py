#!/usr/bin/env python3
"""Compares two Chemical Checker bioactivity spaces (e.g. a space rebuilt
with additional data against its earlier version) using the diagnosis
artifacts already written by ``signature.diagnosis().canvas()`` plus a
shared-compound nearest-neighbor recapitulation test (Comajuncosa-Creus et
al., Nat. Protoc. 2025, Fig. 3 / Extended Data Fig. 8).

All comparison logic lives in ``cc_compare.py``; this script only wires
together CLI configuration and calls ``run_comparison()``. It is meant to be
run after both spaces have already been fitted and diagnosed (e.g. via
``pertprot_cli.py`` or an equivalent pipeline script) -- it does not refit
any signatures, it only reads existing sign3 signatures and diagnosis
artifacts from disk.

Example
-------
python compare_cc_spaces.py \\
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

# Make cc_compare.py / utils.py (placed alongside this script) importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from cc_compare import run_comparison
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
        help="Cosine-distance percentile used to define nearest neighbors in the "
        "shared-key recapitulation test (default: 0.01, as in the CC Protocols paper).",
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
        "--random-state", type=int, default=None, help="Random seed for reproducible subsampling."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing comparison figures (confidence overlay, across-CC scatter); "
        "tables and the recapitulation JSON are always written.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="DEBUG",
        choices=["CRITICAL", "ERROR", "WARN", "INFO", "DEBUG"],
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
        The dictionary returned by :func:`cc_compare.run_comparison`
        (summary table, across-CC comparison, recapitulation results).
    """
    args = parse_args(argv)

    setup_logging(generate_log_filename(args.log_dir, suffix=get_basename(__file__)))

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
        from chemicalchecker import ChemicalChecker

        ChemicalChecker.set_verbosity(args.verbosity)

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
            make_plots=not args.no_plots,
        )

        logger.info("Comparison complete. Outputs written to %s", args.output_dir)

    return results


if __name__ == "__main__":
    main()