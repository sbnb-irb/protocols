"""Archive a finished experiment folder: slim its CC instances, back up its small files and move it.

An experiment is one folder ``<YYYY-MM>_<name>/`` with an ``experiment.yaml`` next to its
configs, scripts, results and Chemical Checker (CC) instances (folders that contain ``full/``).
Archiving it

1. deletes the regenerable files of its instances (see :mod:`chemcheck_protocols.archiving`),
2. writes an "essentials backup", a ``.tar.gz`` of everything except the CC instances, to a
   storage that is not purged,
3. marks ``experiment.yaml`` as archived and moves the folder to the archive root,
4. adds a row to the table in the archive root's ``README.md``.

Where the results, archive and backup roots are is site-specific: options or the environment
variables in :data:`ROOT_VARIABLES`. :func:`plan_archive` checks everything and changes
nothing; :func:`apply_archive` carries the plan out.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints

from .archiving import delete_files, find_regenerable_files, format_size

logger = logging.getLogger(__name__)

ROOT_VARIABLES = {
    "results": "CC_RESULTS_ROOT",
    "archive": "CC_ARCHIVE_ROOT",
    "backup": "CC_BACKUP_ROOT",
}
TEXT_SUFFIXES = {".yaml", ".yml", ".py", ".sh", ".json", ".md", ".txt", ".cfg"}
MAX_TEXT_FILE_BYTES = 1024 * 1024

# A year and month, optionally with the day: YAML turns 2026-09-23 into a date, 2026-09 stays text.
PartialDate = Annotated[
    str,
    BeforeValidator(str),
    StringConstraints(pattern=r"^\d{4}-\d{2}(-\d{2})?$"),
]


class ExperimentRecord(BaseModel):
    """
    The ``experiment.yaml`` of an experiment folder.

    Parameters
    ----------
    name : str
        Short name; the folder is ``<YYYY-MM of started>_<name>``.
    question : str
        What the experiment asked; used as its description in the archive README.
    status : {"active", "analysed", "archived"}
        Where the experiment stands.
    started, finished, archived : str, optional
        ``YYYY-MM`` or ``YYYY-MM-DD``.
    summary : str
        Where the summary is (a notes file, slides, a document).
    layout : str, optional
        What the folder contains and how it is organised.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    question: str
    status: Literal["active", "analysed", "archived"]
    started: PartialDate
    finished: PartialDate | None = None
    archived: PartialDate | None = None
    summary: str
    layout: str | None = None


@dataclass
class ArchivePlan:
    """Everything :func:`apply_archive` will do, and the problems found while planning."""

    experiment: Path
    record: ExperimentRecord
    destination: Path
    backup_file: Path
    readme: Path
    instances: list[Path]
    regenerable_files: list[Path]
    links_that_break: list[Path]
    files_mentioning_path: list[Path]


def root_from_option_or_environment(option: Path | None, kind: str) -> Path:
    """
    Return the ``kind`` root (``results``, ``archive`` or ``backup``): the option, else its variable.

    Raises
    ------
    ValueError
        If neither is given, or the folder does not exist.
    """
    variable = ROOT_VARIABLES[kind]
    root = option or (Path(os.environ[variable]) if os.environ.get(variable) else None)
    if root is None:
        raise ValueError(
            f"No {kind} root: give --{kind}-root or set {variable} (see sbnb.env)"
        )
    if not root.is_dir():
        raise ValueError(f"The {kind} root {root} is not a folder")
    return root


def resolve_experiment(experiment: str, results_root: Path | None) -> Path:
    """
    Find an experiment folder given as an existing path or as a name inside the results root.

    Raises
    ------
    ValueError
        If neither interpretation is a folder.
    """
    as_path = Path(experiment)
    if as_path.is_dir():
        return as_path.resolve()
    if results_root is not None and (results_root / experiment).is_dir():
        return (results_root / experiment).resolve()
    where = (
        f" or in {results_root}" if results_root else " (no results root to look in)"
    )
    raise ValueError(f"No experiment folder {experiment!r}: not a path{where}")


def load_experiment_record(experiment: Path) -> ExperimentRecord:
    """
    Read and validate ``<experiment>/experiment.yaml``.

    Raises
    ------
    ValueError
        If the file is missing or invalid, or the folder is not named
        ``<YYYY-MM of started>_<name>``.
    """
    record_file = experiment / "experiment.yaml"
    if not record_file.is_file():
        raise ValueError(f"{experiment} has no experiment.yaml")
    record = ExperimentRecord.model_validate(yaml.safe_load(record_file.read_text()))
    expected_folder = f"{record.started[:7]}_{record.name}"
    if experiment.name != expected_folder:
        raise ValueError(
            f"Folder {experiment.name!r} does not match experiment.yaml: expected "
            f"{expected_folder!r} (started {record.started}, name {record.name})"
        )
    return record


def find_instances(experiment: Path) -> list[Path]:
    """List the CC instances in an experiment: folders with a ``full/`` subfolder, links not followed."""
    instances = []
    for folder, subfolders, _ in os.walk(experiment):
        if "full" in subfolders and (Path(folder) / "full").is_dir():
            instances.append(Path(folder))
            subfolders.clear()
    return sorted(instances)


def find_references_to(experiment: Path) -> tuple[list[Path], list[Path]]:
    """
    Find what would stop pointing at the experiment once it is moved.

    Walks the folder that contains the experiment (the other experiments too), without
    following links.

    Returns
    -------
    links_that_break : list of pathlib.Path
        Symbolic links that lead into the experiment and either sit outside it or are absolute.
    files_mentioning_path : list of pathlib.Path
        Small text files (configs, scripts, notes) that contain the experiment's absolute path.
    """
    real_experiment = experiment.resolve()
    needles = {str(experiment), str(real_experiment)}
    links_that_break, files_mentioning_path = [], []
    for folder, subfolders, filenames in os.walk(experiment.parent):
        for name in subfolders + filenames:
            path = Path(folder) / name
            if path.is_symlink():
                target = Path(os.path.realpath(path))
                if target.is_relative_to(real_experiment) and (
                    not path.is_relative_to(experiment)
                    or os.path.isabs(os.readlink(path))
                ):
                    links_that_break.append(path)
        for name in filenames:
            path = Path(folder) / name
            if (
                path.suffix in TEXT_SUFFIXES
                and not path.is_symlink()
                and path.stat().st_size <= MAX_TEXT_FILE_BYTES
            ):
                text = path.read_text(errors="ignore")
                if any(needle in text for needle in needles):
                    files_mentioning_path.append(path)
    return sorted(links_that_break), sorted(files_mentioning_path)


def _table_end(readme_lines: list[str]) -> int:
    """Index just after the last row of the README's experiments table (header starts ``| Folder``)."""
    header = next(
        (i for i, line in enumerate(readme_lines) if line.startswith("| Folder")), None
    )
    if header is None:
        raise ValueError(
            "No experiments table (a header line '| Folder | ...') in the README"
        )
    end = header + 1
    while end < len(readme_lines) and readme_lines[end].startswith("|"):
        end += 1
    return end


def plan_archive(
    experiment: Path, archive_root: Path, backup_root: Path
) -> ArchivePlan:
    """
    Check that ``experiment`` can be archived and collect what archiving will do. Changes nothing.

    Raises
    ------
    ValueError
        If ``experiment.yaml`` is missing, invalid or already marked archived, if the
        destination or the backup file exists, or if the archive README has no experiments table.
    """
    record = load_experiment_record(experiment)
    if record.status == "archived":
        raise ValueError(f"{experiment.name} is already marked archived")
    destination = archive_root / experiment.name
    backup_file = backup_root / f"{experiment.name}_essentials.tar.gz"
    for path in (destination, backup_file):
        if path.exists():
            raise ValueError(f"{path} already exists")
    readme = archive_root / "README.md"
    if not readme.is_file():
        raise ValueError(f"No README.md in the archive root {archive_root}")
    _table_end(readme.read_text().splitlines())

    instances = find_instances(experiment)
    regenerable_files = [
        path for instance in instances for path in find_regenerable_files(instance)
    ]
    links_that_break, files_mentioning_path = find_references_to(experiment)
    return ArchivePlan(
        experiment=experiment,
        record=record,
        destination=destination,
        backup_file=backup_file,
        readme=readme,
        instances=instances,
        regenerable_files=regenerable_files,
        links_that_break=links_that_break,
        files_mentioning_path=files_mentioning_path,
    )


def write_essentials_backup(
    experiment: Path, instances: list[Path], backup_file: Path
) -> None:
    """Write ``experiment`` without its CC instances to a ``.tar.gz``; links are stored, not followed."""
    skipped = {
        f"{experiment.name}/{instance.relative_to(experiment)}"
        for instance in instances
    }
    with tarfile.open(backup_file, "w:gz") as archive:
        archive.add(
            experiment,
            arcname=experiment.name,
            filter=lambda member: None if member.name in skipped else member,
        )
    logger.info("Wrote %s (%s)", backup_file, format_size(backup_file.stat().st_size))


def total_size(folder: Path) -> int:
    """Sum of the sizes of the files in ``folder`` (links counted as links, not followed)."""
    return sum(
        (Path(parent) / name).lstat().st_size
        for parent, _, filenames in os.walk(folder)
        for name in filenames
    )


def _readme_row(folder_name: str, record: ExperimentRecord, size_bytes: int) -> str:
    finished = record.finished or record.started
    dates = (
        record.started
        if finished == record.started
        else f"{record.started} to {finished}"
    )
    question = re.sub(r"\s*\n\s*", " ", record.question.strip()).replace("|", r"\|")
    return f"| `{folder_name}/` | {question} | {dates} | {format_size(size_bytes)} |"


def apply_archive(plan: ArchivePlan, today: date | None = None) -> None:
    """
    Carry out ``plan``: prune the instances, back up, mark archived, move and add the README row.

    The backup is written before anything is deleted.
    """
    today = today or datetime.now().astimezone().date()
    write_essentials_backup(plan.experiment, plan.instances, plan.backup_file)
    freed_bytes = delete_files(plan.regenerable_files)
    logger.info(
        "Pruned %d files, freed %s",
        len(plan.regenerable_files),
        format_size(freed_bytes),
    )

    record_file = plan.experiment / "experiment.yaml"
    raw_record = yaml.safe_load(record_file.read_text())
    raw_record.update(status="archived", archived=today)
    record_file.write_text(
        yaml.safe_dump(raw_record, sort_keys=False, allow_unicode=True)
    )

    shutil.move(plan.experiment, plan.destination)
    logger.info("Moved %s to %s", plan.experiment, plan.destination)

    readme_lines = plan.readme.read_text().splitlines()
    row = _readme_row(plan.experiment.name, plan.record, total_size(plan.destination))
    readme_lines.insert(_table_end(readme_lines), row)
    plan.readme.write_text("\n".join(readme_lines) + "\n")
    logger.info("Added the experiment to %s", plan.readme)
