import numpy as np
import pytest

from chem_checker_protocols.config import DatasetConfig
from chem_checker_protocols.signature_fitting import (
    build_reference_sign2_spaces,
    get_cc_universe,
    run_signature_pipeline,
)


class FakeSignature:
    """Records fit() calls; mimics the chemicalchecker methods the pipeline uses."""

    def __init__(self, cctype, dataset_code, keys=(), available=True):
        self.cctype, self.dataset_code = cctype, dataset_code
        self.keys, self._available = list(keys), available
        self.fit_calls = []

    def clear_all(self):
        pass

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))

    def available(self):
        return self._available

    def __array__(self, dtype=None, copy=None):
        return np.zeros((2, 2))


class FakeCC:
    def __init__(self, missing_sign2=()):
        self.signatures, self.missing_sign2 = {}, set(missing_sign2)

    def datasets_exemplary(self):
        return (f"{level}{number}.001" for level in "ABCDE" for number in "12345")

    def get_signature(self, cctype, molset, dataset_code):
        key = (cctype, dataset_code)
        if key not in self.signatures:
            keys = [f"{dataset_code}-mol"]
            self.signatures[key] = FakeSignature(
                cctype, dataset_code, keys, dataset_code not in self.missing_sign2
            )
        return self.signatures[key]

    def signature(self, dataset_code, cctype):
        return self.get_signature(cctype, "full", dataset_code)


def dataset_config(tmp_path, **overrides):
    matrix_file = tmp_path / "raw.csv"
    matrix_file.write_text("inchikey,f1\nAAAA-X,1\n")
    return DatasetConfig.model_validate(
        {
            "key": "m1",
            "name": "M1",
            "dataset_code": "M1.001",
            "source": {"format": "wide_matrix", "path": str(matrix_file)},
            **overrides,
        }
    )


def test_new_space_is_appended_to_the_25_exemplary_spaces():
    spaces = build_reference_sign2_spaces(FakeCC(), "M1.001")
    assert len(spaces) == 26
    assert spaces[-1].dataset_code == "M1.001"


def test_extended_space_replaces_its_original_in_place():
    codes = [
        s.dataset_code
        for s in build_reference_sign2_spaces(FakeCC(), "B1.002", extends="B1.001")
    ]
    assert len(codes) == 25
    assert "B1.001" not in codes
    assert codes.index("B1.002") == 5


def test_extending_a_non_exemplary_space_is_rejected():
    with pytest.raises(ValueError, match="not an exemplary CC space"):
        build_reference_sign2_spaces(FakeCC(), "M1.002", extends="M1.001")


def test_cc_universe_skips_spaces_without_sign2():
    universe = get_cc_universe(FakeCC(missing_sign2={"A2.001"}))
    assert len(universe) == 24
    assert "A2.001-mol" not in universe


@pytest.mark.parametrize(
    ("max_stage", "fitted"),
    [
        ("sign0", ["sign0"]),
        ("sign2", ["sign0", "sign1", "neig1", "sign2"]),
    ],
)
def test_pipeline_stops_after_max_stage(tmp_path, max_stage, fitted):
    result = run_signature_pipeline(
        FakeCC(), dataset_config(tmp_path), diagnosis_plots=False, max_stage=max_stage
    )
    assert list(result) == fitted


def test_fit_options_reach_each_signature_fit(tmp_path):
    config = dataset_config(
        tmp_path,
        fit={
            "sign0": {"sanitizer_kwargs": {"max_features": 10}},
            "sign3": {"complete_universe": "full"},
        },
    )
    result = run_signature_pipeline(FakeCC(), config, diagnosis_plots=False)
    _, sign0_kwargs = result["sign0"].fit_calls[0]
    reference_spaces, sign3_kwargs = result["sign3"].fit_calls[0]
    assert sign0_kwargs["sanitizer_kwargs"] == {"max_features": 10}
    assert sign0_kwargs["keys"] == ["AAAA-X"]
    assert sign3_kwargs["complete_universe"] == "full"
    assert len(reference_spaces[0]) == 26
