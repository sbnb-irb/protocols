"""Run and dataset configuration for signature fitting, read from YAML and validated with pydantic.

A run configuration names the local Chemical Checker (CC) instance and lists the
datasets to signaturize. Relative paths in the file are resolved against the
folder that contains it, so a configuration works from any working directory.
See ``configs/paper_tasks/`` for examples.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

logger = logging.getLogger(__name__)

PipelineStage = Literal["sign0", "sign1", "sign2", "sign3"]
PIPELINE_STAGES: tuple[str, ...] = get_args(PipelineStage)

# CC dataset codes: coordinate (level letter + digit) and a three-digit version, e.g. "B1.002".
DATASET_CODE_PATTERN = r"^[A-Z][0-9]\.[0-9]{3}$"


def check_stage_range(start_stage: str, max_stage: str) -> None:
    """
    Check that ``start_stage`` does not come after ``max_stage`` in :data:`PIPELINE_STAGES`.

    Raises
    ------
    ValueError
        If either is not a pipeline stage, or ``start_stage`` comes after ``max_stage``.
    """
    for name, stage in (("start_stage", start_stage), ("max_stage", max_stage)):
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"{name} must be one of {PIPELINE_STAGES}, got {stage!r}")
    if PIPELINE_STAGES.index(start_stage) > PIPELINE_STAGES.index(max_stage):
        raise ValueError(
            f"start_stage {start_stage!r} comes after max_stage {max_stage!r}"
        )


def _resolve_against_config_dir(path: Path, info: ValidationInfo) -> Path:
    """Resolve a relative path against the config file's folder, when known."""
    base_dir = (info.context or {}).get("config_dir")
    if base_dir is None or path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


ConfigPath = Annotated[Path, AfterValidator(_resolve_against_config_dir)]


class _ConfigModel(BaseModel):
    """Base for every configuration section: unknown keys are errors, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class DataSource(_ConfigModel):
    """
    Where a dataset's raw bioactivity data lives and in which format.

    Parameters
    ----------
    format : {"wide_matrix", "long_pairs", "cc_h5"}
        ``wide_matrix``: delimited text, one row per compound (InChIKeys in the
        first column), one column per feature. ``long_pairs``: an HDF5 file with
        a ``pairs`` dataset of (InChIKey, feature[, value]) rows, as accepted by
        ``sign0.fit(pairs=...)``. ``cc_h5``: an HDF5 file with ``X``, ``keys``
        and ``features`` datasets, as accepted by ``sign0.fit(data_file=...)``.
    path : pathlib.Path
        The data file.
    separator : str, default ","
        Column separator for ``wide_matrix`` files (e.g. ``"\\t"`` for TSV).
    """

    format: Literal["wide_matrix", "long_pairs", "cc_h5"]
    path: ConfigPath
    separator: str = ","


class FitOptions(_ConfigModel):
    """
    Keyword arguments forwarded to each signature type's ``fit()``.

    Keys use the chemicalchecker API names, e.g. ``sign0: {sanitizer_kwargs:
    {max_features: 13000}}`` or ``sign3: {complete_universe: full}``.
    """

    sign0: dict[str, Any] = Field(default_factory=dict)
    sign1: dict[str, Any] = Field(default_factory=dict)
    sign2: dict[str, Any] = Field(default_factory=dict)
    sign3: dict[str, Any] = Field(default_factory=dict)


class BinarizationConfig(_ConfigModel):
    """
    How continuous values become binary up/down profiles for the triplet sampler.

    Each input column gives an "up" and a "down" feature, and missing values are never
    called. With ``log2fc`` and/or ``zscore``, a value is called when it passes either
    rule (the lab's DEP rule for DeepCoverMOA is ``{log2fc: 1.0, zscore: 5.0}``); the
    z-score is per column, over the molecules being signaturized (sample SD). With
    ``percentile``, the cut-off is that percentile of all values: above it is "up",
    below its negative is "down" (the M2 notebooks' scheme, with both sides computed
    from the original values).

    Parameters
    ----------
    log2fc : float, optional
        Absolute value at or above which a value is called.
    zscore : float, optional
        Absolute per-column z-score at or above which a value is called.
    percentile : float, optional
        Percentile of all values used as the cut-off; not combined with the others.
    """

    log2fc: float | None = Field(default=None, gt=0)
    zscore: float | None = Field(default=None, gt=0)
    percentile: float | None = Field(default=None, gt=0, lt=100)

    @model_validator(mode="after")
    def _one_kind_of_rule(self) -> BinarizationConfig:
        if self.percentile is None and self.log2fc is None and self.zscore is None:
            raise ValueError("binarize needs log2fc, zscore or percentile")
        if self.percentile is not None and (self.log2fc or self.zscore):
            raise ValueError("binarize: use percentile alone, or log2fc/zscore")
        return self


class TripletSamplerConfig(_ConfigModel):
    """
    A non-default sampler for the triplets sign3 is trained on.

    Parameters
    ----------
    method : {"bin_jaccard"}
        ``bin_jaccard``: chemicalchecker's ``BinJaccardTripletSampler``, positives
        from the Jaccard similarity of binary profiles.
    triplet_signature : {"raw", "sign0", "sign1"}, default "sign0"
        What defines similar molecules: the dataset's raw input (``wide_matrix``
        sources; held-out molecules removed, missing values kept) or one of its
        fitted signatures. Must be binary unless ``binarize`` is given.
    binarize : BinarizationConfig, optional
        Turn continuous values into up/down profiles first.
    options : dict
        Forwarded to the sampler's ``generate_triplets``, e.g. ``{seed: 0}``.
    """

    method: Literal["bin_jaccard"]
    triplet_signature: Literal["raw", "sign0", "sign1"] = "sign0"
    binarize: BinarizationConfig | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ExtendedSpace(_ConfigModel):
    """The existing CC space a dataset extends (its sign2 replaces that space's in sign3 training)."""

    extends: str = Field(pattern=DATASET_CODE_PATTERN)


class DatasetConfig(_ConfigModel):
    """
    One dataset to signaturize.

    Parameters
    ----------
    key : str
        Short identifier used in logs and in ``--datasets``.
    name : str
        Display name used in log messages.
    dataset_code : str
        CC dataset code the signatures are written to, e.g. ``"M1.001"``.
    source : DataSource
        The raw data to fit sign0 from.
    fit : FitOptions
        Extra ``fit()`` arguments per signature type.
    reference_spaces : "new_space" or ExtendedSpace, default "new_space"
        Which sign2 spaces train sign3: the 25 exemplary CC spaces plus this
        one (``new_space``, paper Tasks 3-4), or the 25 with the extended
        space swapped for this one (``{extends: B1.001}``, paper Tasks 1-2).
    triplet_sampler : TripletSamplerConfig, optional
        Sampler for sign3's training triplets; chemicalchecker's default
        (neighbours in sign1) when omitted.
    holdout_keys : pathlib.Path, optional
        Text file of InChIKeys (one per line) left out of the space before
        sign0, to evaluate sign3 on unseen molecules. Use the same file in the
        runs you compare. ``wide_matrix`` sources only.
    """

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dataset_code: str = Field(pattern=DATASET_CODE_PATTERN)
    source: DataSource
    fit: FitOptions = Field(default_factory=FitOptions)
    reference_spaces: Literal["new_space"] | ExtendedSpace = "new_space"
    triplet_sampler: TripletSamplerConfig | None = None
    holdout_keys: ConfigPath | None = None

    @model_validator(mode="after")
    def _holdout_needs_wide_matrix(self) -> DatasetConfig:
        if self.holdout_keys is not None and self.source.format != "wide_matrix":
            raise ValueError(
                f"holdout_keys needs a wide_matrix source, not {self.source.format}"
            )
        return self

    @model_validator(mode="after")
    def _raw_triplets_need_wide_matrix(self) -> DatasetConfig:
        sampler = self.triplet_sampler
        if (
            sampler is not None
            and sampler.triplet_signature == "raw"
            and self.source.format != "wide_matrix"
        ):
            raise ValueError(
                f"triplet_signature raw needs a wide_matrix source, not {self.source.format}"
            )
        return self

    @model_validator(mode="after")
    def _triplets_set_in_one_place(self) -> DatasetConfig:
        conflicting = sorted({"triplets_sampler", "triplet_sign"} & set(self.fit.sign3))
        if conflicting:
            raise ValueError(
                f"Set the sign3 triplets with triplet_sampler, not fit.sign3 {conflicting}"
            )
        return self

    @model_validator(mode="after")
    def _extension_matches_coordinate(self) -> DatasetConfig:
        if isinstance(self.reference_spaces, ExtendedSpace):
            extended = self.reference_spaces.extends
            if extended[:2] != self.dataset_code[:2] or extended == self.dataset_code:
                raise ValueError(
                    f"{self.dataset_code} cannot extend {extended}: it must be a different "
                    f"version of the same coordinate, e.g. {self.dataset_code[:2]}.001"
                )
        return self


class RunConfig(_ConfigModel):
    """
    Everything one signature-fitting run needs.

    Parameters
    ----------
    cc_root : pathlib.Path
        Local CC instance the signatures are written to (created if missing).
    cc_config : pathlib.Path, optional
        chemicalchecker ``cc_config.json``; if omitted, the ``CC_CONFIG``
        environment variable must be set.
    custom_data_path : pathlib.Path, optional
        Folder with the downloaded reference CC signatures, forwarded to
        ``ChemicalChecker(custom_data_path=...)`` (paper Procedure step 3).
    inchikey_mapping : pathlib.Path, optional
        JSON file mapping InChIKey -> InChI, forwarded to ``sign3.fit`` as
        ``mapping_dict`` so it doesn't query online repositories.
    log_dir : pathlib.Path, default "logs"
        Folder for the run's log file (next to the configuration file by default).
    start_stage : {"sign0", "sign1", "sign2", "sign3"}, default "sign0"
        First signature type to fit; earlier ones are loaded from ``cc_root``
        (e.g. "sign3" to fit only sign3 on an already checked sign2).
    max_stage : {"sign0", "sign1", "sign2", "sign3"}, default "sign3"
        Last signature type to fit.
    diagnosis_plots : bool, default True
        Save CC diagnosis canvases after each stage.
    cc_verbosity : str, default "INFO"
        Log level for chemicalchecker's own logger.
    datasets : list of DatasetConfig
        At least one dataset; keys and dataset codes must be unique.
    """

    cc_root: ConfigPath
    cc_config: ConfigPath | None = None
    custom_data_path: ConfigPath | None = None
    inchikey_mapping: ConfigPath | None = None
    log_dir: ConfigPath = Field(default=Path("logs"), validate_default=True)
    start_stage: PipelineStage = "sign0"
    max_stage: PipelineStage = "sign3"
    diagnosis_plots: bool = True
    cc_verbosity: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    datasets: list[DatasetConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_datasets(self) -> RunConfig:
        for field_name in ("key", "dataset_code"):
            values = [getattr(dataset, field_name) for dataset in self.datasets]
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                raise ValueError(f"Duplicate dataset {field_name}(s): {duplicates}")
        check_stage_range(self.start_stage, self.max_stage)
        return self

    def select_datasets(self, keys: list[str] | None) -> list[DatasetConfig]:
        """
        Return the datasets with the given keys, in config order (all when ``keys`` is None).

        Raises
        ------
        ValueError
            If any key is not defined in the configuration.
        """
        if keys is None:
            return list(self.datasets)
        known = {dataset.key for dataset in self.datasets}
        unknown = sorted(set(keys) - known)
        if unknown:
            raise ValueError(
                f"Unknown dataset key(s) {unknown}; defined keys: {sorted(known)}"
            )
        return [dataset for dataset in self.datasets if dataset.key in keys]


def load_run_config(config_path: str | os.PathLike[str]) -> RunConfig:
    """
    Read and validate a run configuration YAML file.

    Parameters
    ----------
    config_path : str or os.PathLike
        The YAML file.

    Returns
    -------
    RunConfig
        The validated configuration, with relative paths resolved against the
        file's folder.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` is not a file.
    ValueError
        If the file is not valid YAML.
    pydantic.ValidationError
        If the content does not match the schema (unknown keys, missing fields,
        malformed dataset codes, duplicates...).
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    config = RunConfig.model_validate(
        raw, context={"config_dir": config_path.resolve().parent}
    )
    logger.info(
        "Loaded run configuration %s: %d dataset(s)", config_path, len(config.datasets)
    )
    return config


DatasetCode = Annotated[str, Field(pattern=DATASET_CODE_PATTERN)]


class EvaluationReference(_ConfigModel):
    """
    What defines the true neighbours in an evaluation: a CC signature, or profiles from a file.

    Parameters
    ----------
    name : str
        Label used in the output tables.
    signature : {"sign0", "sign1", "sign2", "sign3"}, optional
        A signature of ``cc_root`` / ``dataset_code`` (e.g. sign0 of the evaluated space, or
        sign2 of B1.001 for MoA).
    dataset_code : str, optional
        Dataset of the signature; defaults to the evaluation's ``dataset_code``.
    cc_root : pathlib.Path, optional
        CC instance holding the signature; defaults to the evaluation's ``reference_cc``.
    profiles : DataSource, optional
        A ``wide_matrix`` file instead of a signature (duplicated InChIKeys are averaged).
    binarize : BinarizationConfig, optional
        Turn the profiles into up/down calls first (e.g. the DEP rule).
    """

    name: str
    signature: PipelineStage | None = None
    dataset_code: DatasetCode | None = None
    cc_root: ConfigPath | None = None
    profiles: DataSource | None = None
    binarize: BinarizationConfig | None = None

    @model_validator(mode="after")
    def _signature_or_profiles(self) -> EvaluationReference:
        if (self.signature is None) == (self.profiles is None):
            raise ValueError(
                f"reference {self.name!r}: give either signature or profiles"
            )
        if self.profiles is None and (self.binarize is not None):
            raise ValueError(
                f"reference {self.name!r}: binarize applies to profiles only"
            )
        if self.profiles is not None and self.profiles.format != "wide_matrix":
            raise ValueError(
                f"reference {self.name!r}: profiles must be a wide_matrix file"
            )
        return self


class FoldModel(_ConfigModel):
    """One fold of a run: a CC instance fitted without the molecules in ``holdout``."""

    cc_root: ConfigPath
    holdout: ConfigPath


class EvaluationRun(_ConfigModel):
    """
    A model to score: one CC instance, or a set of fold models pooled into one result.

    Parameters
    ----------
    name : str
        Label used in the output tables.
    cc_root : pathlib.Path, optional
        CC instance with the sign3 to score.
    dataset_code : str, optional
        Dataset of that sign3; defaults to the evaluation's ``dataset_code``.
    folds : list of FoldModel, optional
        Instead of ``cc_root``: each held-out molecule is scored with the fold model that
        never saw it.
    """

    name: str
    cc_root: ConfigPath | None = None
    dataset_code: DatasetCode | None = None
    folds: list[FoldModel] | None = Field(default=None, min_length=2)

    @model_validator(mode="after")
    def _instance_or_folds(self) -> EvaluationRun:
        if (self.cc_root is None) == (self.folds is None):
            raise ValueError(f"run {self.name!r}: give either cc_root or folds")
        return self


class EvaluationConfig(_ConfigModel):
    """
    Held-out evaluation of several runs against one or more references.

    Each held-out molecule is paired with every other molecule that is in the evaluated
    space (sign0 of ``reference_cc`` / ``dataset_code``), has a sign3 in every run and a
    vector in the reference; pairs closer than the background ``p_value_cutoff`` cutoff in
    the reference are the true neighbours, and each run is scored by AUROC on ranking them.
    All runs share the bootstrap resamples, so differences are paired.

    Parameters
    ----------
    reference_cc : pathlib.Path
        CC instance fitted on all molecules (held-out ones included).
    dataset_code : str
        The evaluated space, e.g. ``"D6.002"``.
    holdout : pathlib.Path, optional
        Held-out molecules (``load_key_list`` format). Not needed when a run has ``folds``:
        the folds' held-out files are then the held-out set, in fold order.
    runs : list of EvaluationRun
        The models to score; names must be unique.
    references : list of EvaluationReference
        At least one; names must be unique.
    baseline : str, optional
        Run every other run is compared with; defaults to the first run.
    comparisons : list of [str, str]
        Extra paired differences ``first - second`` between runs.
    n_bootstrap : int, default 100
        Resamples of the held-out molecules.
    seed : int, default 0
        Seed for the background cutoff and the resamples.
    p_value_cutoff : float, default 0.01
        Background-distribution p-value defining the neighbour cutoff.
    per_molecule : bool, default False
        Also write each held-out molecule's own AUROC (single-instance runs only).
    cc_diagnostics : bool, default False
        Also collect CC's own validation numbers of each single-instance run's sign3.
    output : pathlib.Path
        Output prefix: ``<output>_scores.csv``, ``<output>_comparisons.csv`` and, if
        requested, ``<output>_per_molecule.csv`` and ``<output>_cc_diagnostics.csv``.
    cc_config : pathlib.Path, optional
        chemicalchecker ``cc_config.json``; if omitted, ``CC_CONFIG`` must be set.
    log_dir : pathlib.Path, default "logs"
        Folder for the run's log file.
    """

    reference_cc: ConfigPath
    dataset_code: DatasetCode
    holdout: ConfigPath | None = None
    runs: list[EvaluationRun] = Field(min_length=1)
    references: list[EvaluationReference] = Field(min_length=1)
    baseline: str | None = None
    comparisons: list[tuple[str, str]] = Field(default_factory=list)
    n_bootstrap: int = Field(default=100, ge=10)
    seed: int = 0
    p_value_cutoff: float = Field(default=0.01, gt=0, lt=1)
    per_molecule: bool = False
    cc_diagnostics: bool = False
    output: ConfigPath
    cc_config: ConfigPath | None = None
    log_dir: ConfigPath = Field(default=Path("logs"), validate_default=True)

    @model_validator(mode="after")
    def _consistent(self) -> EvaluationConfig:
        for label, names in (
            ("run", [r.name for r in self.runs]),
            ("reference", [r.name for r in self.references]),
        ):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(f"Duplicate {label} name(s): {duplicates}")
        known = {run.name for run in self.runs}
        wanted = ([self.baseline] if self.baseline else []) + [
            n for pair in self.comparisons for n in pair
        ]
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise ValueError(f"Unknown run name(s) in baseline/comparisons: {unknown}")
        fold_sets = {
            tuple(f.holdout for f in run.folds) for run in self.runs if run.folds
        }
        if len(fold_sets) > 1:
            raise ValueError(
                "Runs with folds must use the same held-out files, in the same order"
            )
        if self.holdout is None and not fold_sets:
            raise ValueError("holdout is required unless a run has folds")
        return self

    @property
    def baseline_run(self) -> str:
        """The run compared with every other run."""
        return self.baseline or self.runs[0].name


def load_evaluation_config(config_path: str | os.PathLike[str]) -> EvaluationConfig:
    """
    Read and validate an evaluation configuration YAML file.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` is not a file.
    ValueError
        If the file is not valid YAML.
    pydantic.ValidationError
        If the content does not match the schema.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    config = EvaluationConfig.model_validate(
        raw, context={"config_dir": config_path.resolve().parent}
    )
    logger.info(
        "Loaded evaluation configuration %s: %d run(s), %d reference(s)",
        config_path,
        len(config.runs),
        len(config.references),
    )
    return config
