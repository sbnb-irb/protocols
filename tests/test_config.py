from pathlib import Path

import pytest
from pydantic import ValidationError

from chemcheck_protocols.config import RunConfig, load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def minimal_config(**dataset_overrides):
    dataset = {
        "key": "m1",
        "name": "Drug-microbiome",
        "dataset_code": "M1.001",
        "source": {"format": "wide_matrix", "path": "data.csv"},
        **dataset_overrides,
    }
    return {"cc_root": "cc", "datasets": [dataset]}


@pytest.mark.parametrize("name", ["m1_001.yaml", "d6_001.yaml"])
def test_paper_task_configs_load_with_paths_resolved_against_config_dir(name):
    config = load_run_config(REPO_ROOT / "configs" / "paper_tasks" / name)
    assert config.cc_root.is_absolute()
    assert config.datasets[0].source.path.is_relative_to(REPO_ROOT / "data")


def test_relative_paths_resolve_against_config_file_folder(tmp_path):
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "cc_root: cc\ndatasets:\n  - {key: m1, name: M1, dataset_code: M1.001,\n"
        "     source: {format: wide_matrix, path: raw/data.csv}}\n"
    )
    config = load_run_config(config_file)
    assert config.cc_root == tmp_path / "cc"
    assert config.datasets[0].source.path == tmp_path / "raw" / "data.csv"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unexpected_key": 1}, "Extra inputs are not permitted"),
        ({"dataset_code": "M1-001"}, "String should match pattern"),
        ({"source": {"format": "excel", "path": "x"}}, "Input should be"),
        ({"reference_spaces": {"extends": "B1.001"}}, "cannot extend B1.001"),
    ],
)
def test_invalid_dataset_config_is_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(minimal_config(**overrides))


def test_duplicate_dataset_codes_are_rejected():
    raw = minimal_config()
    raw["datasets"].append({**raw["datasets"][0], "key": "m1_again"})
    with pytest.raises(ValidationError, match="Duplicate dataset dataset_code"):
        RunConfig.model_validate(raw)


def test_start_stage_after_max_stage_is_rejected():
    raw = {**minimal_config(), "start_stage": "sign3", "max_stage": "sign1"}
    with pytest.raises(ValidationError, match="comes after max_stage"):
        RunConfig.model_validate(raw)


def test_select_datasets_rejects_unknown_keys():
    config = RunConfig.model_validate(minimal_config())
    assert [d.key for d in config.select_datasets(None)] == ["m1"]
    with pytest.raises(ValueError, match=r"Unknown dataset key\(s\) \['d6'\]"):
        config.select_datasets(["d6"])


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run configuration not found"):
        load_run_config(tmp_path / "absent.yaml")


def test_invalid_yaml_names_the_file(tmp_path):
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("datasets: [a, , b]\n")
    with pytest.raises(ValueError, match="Invalid YAML in .*broken.yaml"):
        load_run_config(config_file)
