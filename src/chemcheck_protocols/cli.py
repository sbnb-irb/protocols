"""Command-line interface: ``python -m chemcheck_protocols <command> ...``.

Commands
--------
fit-signatures
    Fit sign0 -> sign3 for the datasets in a run configuration YAML file.
holdout
    Draw held-out molecules: a random fraction, disjoint folds, or folds stratified by group.
evaluate
    Score runs on held-out molecules against one or more references (evaluation YAML file).
prune-instance
    List, and with ``--apply`` delete, the regenerable files of a fitted CC instance.
archive
    Prune, back up and move a finished experiment folder to the archive root (dry run by default).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .archiving import delete_files, find_regenerable_files, format_size
from .config import (
    PIPELINE_STAGES,
    EvaluationConfig,
    RunConfig,
    check_stage_range,
    load_evaluation_config,
    load_run_config,
)
from .data_loaders import (
    load_inchikey_mapping,
    load_key_list,
    read_table_with_required_columns,
)
from .experiment_archive import (
    apply_archive,
    plan_archive,
    resolve_experiment,
    root_from_option_or_environment,
)
from .holdouts import disjoint_folds, random_holdout, stratified_folds, write_key_list
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
        "--start-stage",
        choices=PIPELINE_STAGES,
        default=None,
        help="First signature type to fit; earlier ones are loaded from cc_root instead of "
        "refitted. Overrides start_stage in the configuration (default: sign0).",
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
    fit.set_defaults(handler=fit_signatures, load_config=_load_fit_config)

    holdout = commands.add_parser(
        "holdout",
        help="Draw held-out molecules (random fraction, disjoint folds or stratified folds).",
        description="Draw held-out molecules from a key list, reproducibly: sorted keys and "
        "numpy.random.default_rng(seed). Give --fraction for one held-out set, or --folds for "
        "disjoint folds (stratified when --groups is given).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    holdout.add_argument(
        "--keys",
        type=Path,
        required=True,
        help="Pool of molecules, one InChIKey per line (# comments allowed).",
    )
    holdout.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[],
        metavar="FILE",
        help="Key lists removed from the pool first (e.g. an earlier fold).",
    )
    mode = holdout.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fraction", type=float, help="Share of the pool to hold out.")
    mode.add_argument("--folds", type=int, help="Number of disjoint folds.")
    holdout.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="CSV with columns inchikey,group: deal each group's keys across the "
        "folds (only with --folds); its keys are the pool, after --keys.",
    )
    holdout.add_argument("--seed", type=int, default=0, help="Seed of the draw.")
    holdout.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (--fraction) or prefix: <prefix>_fold<i>.txt (--folds).",
    )
    holdout.set_defaults(handler=draw_holdout, load_config=None)

    evaluate = commands.add_parser(
        "evaluate",
        help="Score runs on held-out molecules against one or more references.",
        description="Held-out neighbour AUROC of each run against each reference, with paired "
        "bootstrap differences; settings in an evaluation configuration YAML file.",
    )
    evaluate.add_argument(
        "--config", type=Path, required=True, help="Evaluation configuration YAML file."
    )
    evaluate.set_defaults(
        handler=evaluate_runs,
        load_config=lambda args: load_evaluation_config(args.config),
    )

    prune = commands.add_parser(
        "prune-instance",
        help="List (and with --apply delete) the regenerable files of a fitted CC instance.",
        description="Remove sign3/models/all_sign2*.h5, which only 'fit' needs and which it "
        "rebuilds from the reference sign2. Signatures, trained networks and training files "
        "are kept. Dry run unless --apply is given.",
    )
    prune.add_argument(
        "cc_root",
        type=Path,
        help="Root of the CC instance (the folder that contains full/).",
    )
    prune.add_argument(
        "--apply", action="store_true", help="Delete the files instead of listing them."
    )
    prune.set_defaults(handler=prune_instance, load_config=None)

    archive = commands.add_parser(
        "archive",
        help="Archive a finished experiment folder (dry run unless --apply).",
        description="Archive an experiment folder (see experiment_archive.py): prune the "
        "regenerable files of its CC instances, write an essentials backup (everything except "
        "the instances), mark experiment.yaml archived, move the folder to the archive root "
        "and add it to the archive README. The roots come from the options or from "
        "CC_RESULTS_ROOT, CC_ARCHIVE_ROOT and CC_BACKUP_ROOT. Dry run unless --apply is given.",
    )
    archive.add_argument(
        "experiment",
        help="Experiment folder: a path, or a name inside the results root.",
    )
    archive.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Where experiment names are looked up (default: $CC_RESULTS_ROOT).",
    )
    archive.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Where archived experiments go (default: $CC_ARCHIVE_ROOT).",
    )
    archive.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Where the essentials backup is written, outside purged storage "
        "(default: $CC_BACKUP_ROOT).",
    )
    archive.add_argument(
        "--apply",
        action="store_true",
        help="Carry out the archiving instead of reporting the plan.",
    )
    archive.set_defaults(handler=archive_experiment, load_config=None)
    return parser


def _load_fit_config(args: argparse.Namespace) -> RunConfig:
    config = load_run_config(args.config)
    config.select_datasets(args.datasets)
    check_stage_range(
        args.start_stage or config.start_stage, args.max_stage or config.max_stage
    )
    return config


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
    start_stage = args.start_stage or config.start_stage
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
        if config.custom_data_path is not None and not any(
            config.custom_data_path.glob("*.h5")
        ):
            logger.error(
                "No *.h5 signatures in custom_data_path %s; check the path and that it "
                "is bound into the container",
                config.custom_data_path,
            )
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
        # The universe overlap is only reported when sign2 is fitted.
        fits_sign2 = (
            PIPELINE_STAGES.index(start_stage)
            <= PIPELINE_STAGES.index("sign2")
            <= PIPELINE_STAGES.index(max_stage)
        )
        cc_universe = get_cc_universe(cc_instance) if fits_sign2 else None

        for dataset in datasets:
            logger.info(
                "===== Fitting %s (%s) from %s to %s =====",
                dataset.name,
                dataset.dataset_code,
                start_stage,
                max_stage,
            )
            run_signature_pipeline(
                cc_instance,
                dataset,
                mapping_dict=mapping_dict,
                cc_universe=cc_universe,
                diagnosis_plots=diagnosis_plots,
                start_stage=start_stage,
                max_stage=max_stage,
            )
        logger.info("Fitted datasets: %s", [dataset.key for dataset in datasets])
    return 0


def draw_holdout(args: argparse.Namespace, config: None = None) -> int:
    """
    Run the ``holdout`` command.

    Returns
    -------
    int
        Exit code: 0 when the held-out files were written.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    excluded = {key for path in args.exclude for key in load_key_list(path)}
    pool = sorted(set(load_key_list(args.keys)) - excluded)
    source = f"{args.keys}" + (
        f" minus {', '.join(str(p) for p in args.exclude)}" if args.exclude else ""
    )
    if args.fraction is not None:
        if args.groups is not None:
            logger.error("--groups applies to --folds only")
            return 2
        keys = random_holdout(pool, args.fraction, args.seed)
        write_key_list(
            args.output,
            keys,
            f"{args.fraction:g} of {len(pool)} molecules (seed "
            f"{args.seed}) from {source}",
        )
        return 0
    if args.groups is not None:
        table = read_table_with_required_columns(
            args.groups, columns=["inchikey", "group"]
        )
        groups = {
            k: str(g)
            for k, g in zip(table["inchikey"], table["group"])
            if k in set(pool)
        }
        folds = stratified_folds(groups, args.folds, args.seed)
        how = f"stratified by group ({args.groups})"
    else:
        folds = disjoint_folds(pool, args.folds, args.seed)
        how = "disjoint"
    for i, keys in enumerate(folds, start=1):
        write_key_list(
            f"{args.output}_fold{i}.txt",
            keys,
            f"fold {i} of {args.folds}, {how}, seed {args.seed}, from {source}",
        )
    return 0


def evaluate_runs(args: argparse.Namespace, config: EvaluationConfig) -> int:
    """
    Run the ``evaluate`` command.

    Returns
    -------
    int
        Exit code: 0 when the result tables were written.
    """
    log_file = generate_log_filename(config.log_dir, suffix="evaluate")
    setup_logging(log_file)
    with log_run():
        if config.cc_config is not None:
            os.environ["CC_CONFIG"] = str(config.cc_config)
        elif "CC_CONFIG" not in os.environ:
            logger.error("No cc_config in %s and CC_CONFIG is not set", args.config)
            return 2
        from .heldout_evaluation import (
            cc_signature_opener,
            run_evaluation,
            write_evaluation,
        )

        opener = cc_signature_opener()
        resume_logging_after_import(log_file)
        tables = run_evaluation(config, opener)
        write_evaluation(tables, config.output)
        for row in tables["comparisons"].itertuples():
            logger.info(
                "%s: %s - %s = %+.3f (95%% CI %+.3f to %+.3f)",
                row.reference,
                row.first,
                row.second,
                row.difference,
                row.ci_low,
                row.ci_high,
            )
    return 0


def prune_instance(args: argparse.Namespace, config: None = None) -> int:
    """
    Run the ``prune-instance`` command.

    Returns
    -------
    int
        Exit code: 0 when the files were listed or deleted, 2 if ``cc_root`` is not an instance.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    try:
        files = find_regenerable_files(args.cc_root)
    except ValueError as error:
        logger.error("%s", error)
        return 2
    total_bytes = sum(path.stat().st_size for path in files)
    if not args.apply:
        for path in files:
            logger.info("%s  %s", format_size(path.stat().st_size), path)
        logger.info(
            "Dry run: %d files, %s; run again with --apply to delete them",
            len(files),
            format_size(total_bytes),
        )
        return 0
    freed_bytes = delete_files(files)
    logger.info("Deleted %d files, freed %s", len(files), format_size(freed_bytes))
    return 0


def archive_experiment(args: argparse.Namespace, config: None = None) -> int:
    """
    Run the ``archive`` command.

    Returns
    -------
    int
        Exit code: 0 when the plan was reported or carried out, 2 when the experiment cannot
        be archived (bad roots or ``experiment.yaml``, existing destination, links that would
        break).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    try:
        archive_root = root_from_option_or_environment(args.archive_root, "archive")
        backup_root = root_from_option_or_environment(args.backup_root, "backup")
        try:
            results_root = root_from_option_or_environment(args.results_root, "results")
        except ValueError:
            results_root = None  # only needed to look an experiment up by name
        experiment = resolve_experiment(args.experiment, results_root)
        plan = plan_archive(experiment, archive_root, backup_root)
    except ValueError as error:
        logger.error("%s", error)
        return 2

    logger.info("Experiment %s -> %s", plan.experiment, plan.destination)
    logger.info("Essentials backup: %s", plan.backup_file)
    logger.info(
        "Pruning %d files in %d CC instances (%s)",
        len(plan.regenerable_files),
        len(plan.instances),
        format_size(sum(path.stat().st_size for path in plan.regenerable_files)),
    )
    for path in plan.files_mentioning_path:
        logger.warning("Mentions the experiment's absolute path: %s", path)
    for path in plan.links_that_break:
        logger.error("Link that would break when the folder moves: %s", path)
    if plan.links_that_break:
        return 2
    if not args.apply:
        logger.info("Dry run: nothing changed; run again with --apply to archive")
        return 0
    apply_archive(plan)
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
    config = None
    if args.load_config is not None:
        try:
            config = args.load_config(args)
        except (FileNotFoundError, ValidationError, ValueError) as error:
            parser.error(str(error))
    return args.handler(args, config)


if __name__ == "__main__":
    sys.exit(main())
