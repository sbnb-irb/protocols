import tarfile

import pytest
import yaml

from chemcheck_protocols.cli import main
from chemcheck_protocols.experiment_archive import (
    ExperimentRecord,
    resolve_experiment,
    root_from_option_or_environment,
)

README = (
    "# cc_archived\n\n## Experiments\n\n"
    "| Folder | What | Dates | Size |\n|---|---|---|---|\n"
    "| `2026-08_old/` | An earlier one | 2026-08 | 1.0 KiB |\n\n"
    "### 2026-08_old\n\nNotes.\n"
)


@pytest.fixture
def sites(tmp_path, monkeypatch):
    """Results, archive and backup roots with one experiment holding a fitted CC instance."""
    results, archive, backup = (
        tmp_path / name for name in ("results", "archive", "backup")
    )
    for root in (results, archive, backup):
        root.mkdir()
    (archive / "README.md").write_text(README)
    experiment = results / "2026-09_demo"
    models = experiment / "local_CC_A/full/A/A1/A1.001/sign3/models"
    models.mkdir(parents=True)
    for name in ("all_sign2.h5", "all_sign2.complete.h5", "train.h5"):
        (models / name).write_bytes(b"x" * 1000)
    (experiment / "configs").mkdir()
    (experiment / "configs" / "run.yaml").write_text("cc_root: ../local_CC_A\n")
    (experiment / "experiment.yaml").write_text(
        "name: demo\nquestion: Does it work | really?\nstatus: analysed\n"
        "started: 2026-09\nfinished: 2026-09-23\nsummary: NOTES.md\n"
    )
    monkeypatch.setenv("CC_RESULTS_ROOT", str(results))
    monkeypatch.setenv("CC_ARCHIVE_ROOT", str(archive))
    monkeypatch.setenv("CC_BACKUP_ROOT", str(backup))
    return experiment, archive, backup


def test_dry_run_changes_nothing(sites):
    experiment, archive, backup = sites
    assert main(["archive", "2026-09_demo"]) == 0
    assert experiment.is_dir()
    assert len(list(experiment.rglob("all_sign2*.h5"))) == 2
    assert not list(archive.glob("2026-09_demo")) and not list(backup.iterdir())
    assert archive.joinpath("README.md").read_text() == README


def test_apply_prunes_backs_up_moves_and_records(sites):
    experiment, archive, backup = sites
    assert main(["archive", "2026-09_demo", "--apply"]) == 0

    moved = archive / "2026-09_demo"
    assert not experiment.exists()
    models = moved / "local_CC_A/full/A/A1/A1.001/sign3/models"
    assert sorted(path.name for path in models.iterdir()) == ["train.h5"]
    record = ExperimentRecord.model_validate(
        yaml.safe_load((moved / "experiment.yaml").read_text())
    )
    assert record.status == "archived" and record.archived is not None

    with tarfile.open(backup / "2026-09_demo_essentials.tar.gz") as tar:
        names = tar.getnames()
    assert "2026-09_demo/configs/run.yaml" in names
    assert not any("local_CC_A" in name for name in names)

    lines = (archive / "README.md").read_text().splitlines()
    row = lines.index("| `2026-08_old/` | An earlier one | 2026-08 | 1.0 KiB |") + 1
    assert lines[row].startswith(
        "| `2026-09_demo/` | Does it work \\| really? | 2026-09 to 2026-09-23 |"
    )
    assert lines[row + 1] == ""


def test_a_link_from_another_experiment_blocks_the_move(sites, tmp_path):
    experiment, archive, _ = sites
    other = tmp_path / "results" / "2026-09_other"
    other.mkdir()
    (other / "reference").symlink_to(experiment / "configs")
    assert main(["archive", "2026-09_demo", "--apply"]) == 2
    assert experiment.is_dir() and not (archive / "2026-09_demo").exists()


def test_a_note_with_the_absolute_path_only_warns(sites, caplog):
    experiment, archive, _ = sites
    (experiment / "NOTES.md").write_text(f"Data in {experiment}/configs\n")
    assert main(["archive", "2026-09_demo", "--apply"]) == 0
    assert "Mentions the experiment's absolute path" in caplog.text
    assert (archive / "2026-09_demo").is_dir()


@pytest.mark.parametrize(
    ("record_text", "message"),
    [
        (None, "no experiment.yaml"),
        (
            "name: other\nquestion: q\nstatus: active\nstarted: 2026-09\nsummary: s\n",
            "does not match",
        ),
        (
            "name: demo\nquestion: q\nstatus: done\nstarted: 2026-09\nsummary: s\n",
            "status",
        ),
        (
            "name: demo\nquestion: q\nstatus: archived\nstarted: 2026-09\nsummary: s\n",
            "already marked",
        ),
    ],
)
def test_an_invalid_record_is_refused(sites, caplog, record_text, message):
    experiment, _, backup = sites
    record_file = experiment / "experiment.yaml"
    if record_text is None:
        record_file.unlink()
    else:
        record_file.write_text(record_text)
    assert main(["archive", "2026-09_demo", "--apply"]) == 2
    assert message in caplog.text
    assert experiment.is_dir() and not list(backup.iterdir())


def test_an_existing_destination_is_refused(sites, caplog):
    _, archive, _ = sites
    (archive / "2026-09_demo").mkdir()
    assert main(["archive", "2026-09_demo", "--apply"]) == 2
    assert "already exists" in caplog.text


def test_roots_come_from_options_or_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CC_ARCHIVE_ROOT", raising=False)
    with pytest.raises(ValueError, match="CC_ARCHIVE_ROOT"):
        root_from_option_or_environment(None, "archive")
    assert root_from_option_or_environment(tmp_path, "archive") == tmp_path
    monkeypatch.setenv("CC_ARCHIVE_ROOT", str(tmp_path))
    assert root_from_option_or_environment(None, "archive") == tmp_path
    with pytest.raises(ValueError, match="not a folder"):
        root_from_option_or_environment(tmp_path / "missing", "archive")


def test_experiment_is_found_by_path_or_by_name(sites):
    experiment, _, _ = sites
    assert resolve_experiment(str(experiment), None) == experiment
    assert resolve_experiment("2026-09_demo", experiment.parent) == experiment
    with pytest.raises(ValueError, match="No experiment folder"):
        resolve_experiment("2026-09_missing", experiment.parent)
