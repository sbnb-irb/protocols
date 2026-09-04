#!/usr/bin/env python3
"""Runs the PerturbProt Chemical Checker signature pipeline non-interactively.

Fits sign0 -> sign1/neig1 -> sign2 -> sign3 for the PerturbProt datasets and
saves diagnosis plots, meant to be submitted as an HPC batch job (e.g. via
sbatch/srun). All fitting/diagnosis logic lives in ``cc_pipeline.py``; this
script only wires together CLI configuration and the loop over datasets.

Use ``--datasets`` to restrict a run to one or more dataset keys -- e.g. to
submit one SLURM array task per dataset instead of looping over all six in a
single job, since sign3 fitting is computationally demanding.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Make cc_pipeline.py / utils.py (placed alongside this script) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cc_pipeline import (
    DATASET_KEYS,
    default_dataset_specs,
    get_cc_universe,
    load_dataset_dataframes,
    run_full_pipeline,
)
from utils import check_path, generate_log_filename, get_basename, log_run, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the PerturbProt HPC pipeline.

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
        description="Run the Chemical Checker sign0->sign3 pipeline for the PerturbProt datasets."
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
        default=Path("/scratch/sbnb/sayala/protocols/local_CC_D6"),
        help="Local Chemical Checker instance directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/scratch/sbnb/sayala/cc_data/data"),
        help="Custom data path for the local Chemical Checker instance (created if missing).",
    )
    parser.add_argument(
        "--perturbprot-data-dir",
        type=Path,
        default=Path("/scratch/sbnb/sayala/protocols/data/PerturbProt"),
        help="Directory with the wide PerturbProt compound matrices "
             "(see transform_allstudies_to_widedf.py).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=list(DATASET_KEYS),
        metavar="KEY",
        help=f"Subset of dataset keys to run (default: all six -- {', '.join(DATASET_KEYS)}). "
             "Pass a single key per job to split the run across an HPC job array.",
    )
    parser.add_argument(
        "--sanitizer-chunk-size",
        type=int,
        default=500_000,
        help="Chunk size passed to sign0's sanitizer_kwargs (adjust for memory).",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="Optional path to a JSON file mapping InChIKeys to InChIs, "
             "forwarded to sign3.fit() as mapping_dict.",
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
    parser.add_argument(
        "--no-diagnosis-plots",
        action="store_true",
        help="Skip diagnosis().canvas() plotting/saving for every stage (faster, no figures).",
    )
    return parser.parse_args(argv)


def load_mapping_dict(mapping_json: Path | None) -> dict[str, str] | None:
    """
    Load an optional InChIKey -> InChI mapping dict from a JSON file.

    Parameters
    ----------
    mapping_json : Path or None
        Path to the JSON file, or None to skip.

    Returns
    -------
    dict or None
        The loaded mapping, or None if ``mapping_json`` is None.

    Raises
    ------
    FileNotFoundError
        If ``mapping_json`` is given but does not exist.
    """
    if mapping_json is None:
        return None
    if not check_path(mapping_json):
        raise FileNotFoundError(f"Mapping JSON file not found at: {mapping_json}")
    with open(mapping_json, encoding="utf-8") as fh:
        mapping_dict = json.load(fh)
    logger.info("Loaded InChIKey->InChI mapping dict with %d entries from %s", len(mapping_dict), mapping_json)
    return mapping_dict


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """
    Run the sign0->sign3 pipeline for all requested PerturbProt datasets.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments for programmatic testing.

    Returns
    -------
    dict
        Mapping of dataset key -> dict of fitted signature objects (see
        :func:`cc_pipeline.run_full_pipeline`).
    """
    args = parse_args(argv)

    setup_logging(generate_log_filename(args.log_dir, suffix=get_basename(__file__)))

    with log_run():
        # CC_CONFIG must be set before chemicalchecker is imported
        os.environ["CC_CONFIG"] = str(args.cc_config)
        from chemicalchecker import ChemicalChecker

        args.data_path.mkdir(parents=True, exist_ok=True)
        ChemicalChecker.set_verbosity(args.verbosity)
        cc_local = ChemicalChecker(
            str(args.local_cc_dir), dbconnect=False, custom_data_path=str(args.data_path)
        )

        mapping_dict = load_mapping_dict(args.mapping_json)

        specs = default_dataset_specs(args.perturbprot_data_dir)
        if args.datasets:
            specs = [spec for spec in specs if spec.key in args.datasets]
        load_dataset_dataframes(specs)

        cc_universe = get_cc_universe(cc_local)
        sanitizer_kwargs = {"chunk_size": args.sanitizer_chunk_size}

        all_results: dict[str, dict[str, Any]] = {}
        for spec in specs:
            logger.info("===== Running pipeline for %s (%s) =====", spec.name, spec.code)
            all_results[spec.key] = run_full_pipeline(
                cc_local,
                spec,
                sanitizer_kwargs=sanitizer_kwargs,
                mapping_dict=mapping_dict,
                cc_universe=cc_universe,
                plot=not args.no_diagnosis_plots,
            )

        logger.info("All requested datasets processed: %s", list(all_results.keys()))

    return all_results


if __name__ == "__main__":
    main()
