import json
import pickle

import pytest

from chemcheck_protocols.cc_diagnostics import read_cc_diagnostics, signature_folder


def _fake_sign3(root):
    folder = root / "full" / "D" / "D6" / "D6.002" / "sign3"
    diag = folder / "diags" / "D6.002_sign3"
    diag.mkdir(parents=True)
    (folder / "stats").mkdir()
    for name, value in (("moa_roc", 0.76), ("atc_roc", 0.74)):
        (diag / f"{name}.pkl").write_bytes(pickle.dumps({"auc": value}))
    (diag / "across_roc.pkl").write_bytes(
        pickle.dumps({"A1.001": {"auc": 0.6}, "B1.001": {"auc": 0.7}})
    )
    (diag / "confidences.pkl").write_bytes(pickle.dumps({"confidences": [0.2, 0.4]}))
    (folder / "stats" / "validation_stats.json").write_text(
        json.dumps({"moa_auc": 0.78, "atc_auc": 0.73})
    )


def test_cc_diagnostics_are_collected_from_the_files(tmp_path):
    _fake_sign3(tmp_path)
    table = read_cc_diagnostics(tmp_path, "D6.002").set_index(["metric", "cc_space"])[
        "value"
    ]
    assert table[("neighbourhood_auroc", "B1.001")] == pytest.approx(0.76)
    assert table[("across_space_auroc", "A1.001")] == pytest.approx(0.6)
    assert table[("annotation_auroc_atc", "")] == pytest.approx(0.73)
    assert table[("mean_confidence", "")] == pytest.approx(0.3)


def test_missing_signature_folder_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No sign3 of M1.001"):
        signature_folder(tmp_path, "M1.001")
