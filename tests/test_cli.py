from chemcheck_protocols.cli import main


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
