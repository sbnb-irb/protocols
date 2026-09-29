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

MaxStage = Literal["sign0", "sign1", "sign2", "sign3"]
PIPELINE_STAGES: tuple[str, ...] = get_args(MaxStage)

# CC dataset codes: coordinate (level letter + digit) and a three-digit version, e.g. "B1.002".
DATASET_CODE_PATTERN = r"^[A-Z][0-9]\.[0-9]{3}$"


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
    """

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dataset_code: str = Field(pattern=DATASET_CODE_PATTERN)
    source: DataSource
    fit: FitOptions = Field(default_factory=FitOptions)
    reference_spaces: Literal["new_space"] | ExtendedSpace = "new_space"

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
        Folder for the run's log file.
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
    log_dir: ConfigPath = Path("logs")
    max_stage: MaxStage = "sign3"
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
    pydantic.ValidationError
        If the content does not match the schema (unknown keys, missing fields,
        malformed dataset codes, duplicates...).
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = RunConfig.model_validate(
        raw, context={"config_dir": config_path.resolve().parent}
    )
    logger.info(
        "Loaded run configuration %s: %d dataset(s)", config_path, len(config.datasets)
    )
    return config
