import pytest

from chemcheck_protocols.cli import main
from chemcheck_protocols.data_loaders import load_key_list


def test_custom_data_path_without_signatures_stops_before_fitting(tmp_path):
    (tmp_path / "raw.csv").write_text("inchikey,f1\nAAAA-X,1\n")
    (tmp_path / "cc_data").mkdir()
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "cc_root: cc\ncc_config: cc_config.json\ncustom_data_path: cc_data\n"
        "datasets:\n  - {key: m1, name: M1, dataset_code: M1.001,\n"
        "     source: {format: wide_matrix, path: raw.csv}}\n"
    )
    assert main(["fit-signatures", "--config", str(config_file)]) == 2
    (log_file,) = (tmp_path / "logs").glob("*.log")
    assert "No *.h5 signatures in custom_data_path" in log_file.read_text()
    assert not (tmp_path / "cc").exists()


def test_holdout_command_writes_reproducible_folds(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("\n".join(f"KEY{i:02d}" for i in range(20)) + "\n")
    (tmp_path / "groups.csv").write_text(
        "inchikey,group\n"
        + "".join(f"KEY{i:02d},{'a' if i < 12 else 'b'}\n" for i in range(20))
    )
    assert (
        main(
            [
                "holdout",
                "--keys",
                str(keys),
                "--fraction",
                "0.25",
                "--output",
                str(tmp_path / "held.txt"),
            ]
        )
        == 0
    )
    assert len(load_key_list(tmp_path / "held.txt")) == 5
    args = [
        "holdout",
        "--keys",
        str(keys),
        "--folds",
        "4",
        "--groups",
        str(tmp_path / "groups.csv"),
        "--exclude",
        str(tmp_path / "held.txt"),
        "--output",
        str(tmp_path / "fold"),
    ]
    assert main(args) == 0
    folds = [load_key_list(tmp_path / f"fold_fold{i}.txt") for i in range(1, 5)]
    assert sorted(k for f in folds for k in f) == sorted(
        set(keys.read_text().split()) - set(load_key_list(tmp_path / "held.txt"))
    )


def test_evaluate_rejects_an_invalid_configuration(tmp_path, capsys):
    config_file = tmp_path / "eval.yaml"
    config_file.write_text(
        "reference_cc: cc\ndataset_code: D6.002\nruns: []\nreferences: []\noutput: out\n"
    )
    with pytest.raises(SystemExit) as stop:
        main(["evaluate", "--config", str(config_file)])
    assert stop.value.code == 2
    assert "runs" in capsys.readouterr().err
