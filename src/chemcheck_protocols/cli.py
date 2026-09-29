"""Command-line interface: ``python -m chemcheck_protocols <command> ...``.

Commands
--------
fit-signatures
    Fit sign0 -> sign3 for the datasets in a run configuration YAML file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .config import PIPELINE_STAGES, RunConfig, load_run_config
from .data_loaders import load_inchikey_mapping
from .run_logging import (
    generate_log_filename,
    log_run,
    resume_logging_after_import,
    setup_logging,
)
from .signature_fitting import get_cc_universe, run_signature_pipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-parser per command."""
    parser = argparse.ArgumentParser(
        prog="chemcheck_protocols",
        description="Chemical Checker protocols: fit, evaluate and compare bioactivity signature spaces.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    fit = commands.add_parser(
        "fit-signatures",
        help="Fit sign0 -> sign3 for the datasets in a run configuration.",
        description="Fit sign0 -> sign1 (+neig1) -> sign2 -> sign3 for each dataset in a run "
        "configuration YAML file (see configs/paper_tasks/ for examples).",
    )
    fit.add_argument(
        "--config", type=Path, required=True, help="Run configuration YAML file."
    )
    fit.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Only fit these dataset keys from the configuration (default: all of them).",
    )
    fit.add_argument(
        "--max-stage",
        choices=PIPELINE_STAGES,
        default=None,
        help="Last signature type to fit; overrides max_stage in the configuration (default: sign3).",
    )
    fit.add_argument(
        "--no-diagnosis-plots",
        action="store_true",
        help="Skip saving CC diagnosis canvases, overriding diagnosis_plots in the configuration.",
    )
    fit.set_defaults(handler=fit_signatures)
    return parser


def fit_signatures(args: argparse.Namespace, config: RunConfig) -> int:
    """
    Run the ``fit-signatures`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : RunConfig
        The validated run configuration.

    Returns
    -------
    int
        Exit code: 0 when every selected dataset was fitted.
    """
    max_stage = args.max_stage or config.max_stage
    diagnosis_plots = config.diagnosis_plots and not args.no_diagnosis_plots
    datasets = config.select_datasets(args.datasets)

    log_file = generate_log_filename(config.log_dir, suffix="fit_signatures")
    setup_logging(log_file)
    with log_run():
        # chemicalchecker reads CC_CONFIG when it is imported, so set it first.
        if config.cc_config is not None:
            os.environ["CC_CONFIG"] = str(config.cc_config)
        elif "CC_CONFIG" not in os.environ:
            logger.error("No cc_config in %s and CC_CONFIG is not set", args.config)
            return 2
        from chemicalchecker import ChemicalChecker

        ChemicalChecker.set_verbosity(config.cc_verbosity)
        # Importing chemicalchecker disables this package's loggers and replaces
        # the root handlers; restore them (see resume_logging_after_import).
        resume_logging_after_import(log_file)

        custom_data_path = (
            None if config.custom_data_path is None else str(config.custom_data_path)
        )
        cc_instance = ChemicalChecker(
            str(config.cc_root), dbconnect=False, custom_data_path=custom_data_path
        )
        mapping_dict = load_inchikey_mapping(config.inchikey_mapping)
        needs_universe = PIPELINE_STAGES.index(max_stage) >= PIPELINE_STAGES.index(
            "sign2"
        )
        cc_universe = get_cc_universe(cc_instance) if needs_universe else None

        for dataset in datasets:
            logger.info(
                "===== Fitting %s (%s) up to %s =====",
                dataset.name,
                dataset.dataset_code,
                max_stage,
            )
            run_signature_pipeline(
                cc_instance,
                dataset,
                mapping_dict=mapping_dict,
                cc_universe=cc_universe,
                diagnosis_plots=diagnosis_plots,
                max_stage=max_stage,
            )
        logger.info("Fitted datasets: %s", [dataset.key for dataset in datasets])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse arguments, load the run configuration and dispatch to the command.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config)
        config.select_datasets(args.datasets)
    except (FileNotFoundError, ValidationError, ValueError) as error:
        parser.error(str(error))
    return args.handler(args, config)


if __name__ == "__main__":
    sys.exit(main())
