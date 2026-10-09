import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chemcheck_protocols.config import DatasetConfig, TripletSamplerConfig
from chemcheck_protocols.signature_fitting import (
    build_reference_sign2_spaces,
    get_cc_universe,
    resolve_triplet_sampler,
    run_signature_pipeline,
)


class FakeSignature:
    """Records fit() calls; mimics the chemicalchecker methods the pipeline uses."""

    def __init__(self, cctype, dataset_code, keys=(), available=True):
        self.cctype, self.dataset_code = cctype, dataset_code
        self.keys, self._available = list(keys), available
        self.data_path = f"/cc/{dataset_code}/{cctype}.h5"
        self.fit_calls = []

    def clear_all(self):
        pass

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))

    def available(self):
        return self._available

    def diagnosis(self, **kwargs):
        # chemicalchecker raises a bare Exception for missing data files
        raise Exception("Data file A1.001/sign1/sign1.h5 not available.")  # noqa: TRY002

    def __array__(self, dtype=None, copy=None):
        return np.zeros((2, 2))


class FakeCC:
    """Every signature counts as fitted except the (cctype, dataset_code) pairs in ``unfitted``."""

    def __init__(self, unfitted=()):
        self.signatures, self.unfitted = {}, set(unfitted)

    def datasets_exemplary(self):
        return (f"{level}{number}.001" for level in "ABCDE" for number in "12345")

    def get_signature(self, cctype, molset, dataset_code):
        key = (cctype, dataset_code)
        if key not in self.signatures:
            keys = [f"{dataset_code}-mol"]
            self.signatures[key] = FakeSignature(
                cctype, dataset_code, keys, key not in self.unfitted
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


def test_missing_reference_sign2_is_rejected():
    with pytest.raises(FileNotFoundError, match=r"sign2 missing for \['A2.001'\]"):
        build_reference_sign2_spaces(FakeCC(unfitted={("sign2", "A2.001")}), "M1.001")


def test_cc_universe_skips_spaces_without_sign2():
    universe = get_cc_universe(FakeCC(unfitted={("sign2", "A2.001")}))
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


def test_start_stage_loads_its_inputs_instead_of_refitting(tmp_path):
    result = run_signature_pipeline(
        FakeCC(), dataset_config(tmp_path), diagnosis_plots=False, start_stage="sign3"
    )
    assert list(result) == ["sign1", "sign2", "sign3"]
    assert result["sign1"].fit_calls == result["sign2"].fit_calls == []
    (_, sign2, sign1), _ = result["sign3"].fit_calls[0]
    assert (sign2, sign1) == (result["sign2"], result["sign1"])


def test_start_stage_with_unfitted_input_is_rejected(tmp_path):
    cc_instance = FakeCC(unfitted={("neig1", "M1.001")})
    with pytest.raises(FileNotFoundError, match="M1.001 neig1 is not fitted"):
        run_signature_pipeline(
            cc_instance,
            dataset_config(tmp_path),
            start_stage="sign2",
            max_stage="sign2",
        )


def test_start_stage_after_max_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="comes after max_stage"):
        run_signature_pipeline(
            FakeCC(), dataset_config(tmp_path), start_stage="sign3", max_stage="sign2"
        )


class FakeSampler:
    pass


@pytest.fixture
def splitter_module(monkeypatch):
    """Stand-in for chemicalchecker.util.splitter providing the custom sampler."""
    module = SimpleNamespace(BinJaccardTripletSampler=FakeSampler, __file__="fake")
    monkeypatch.setitem(sys.modules, "chemicalchecker.util.splitter", module)
    return module


def test_configured_sampler_and_triplet_signature_reach_sign3(
    tmp_path, splitter_module
):
    config = dataset_config(
        tmp_path,
        triplet_sampler={"method": "bin_jaccard", "options": {"seed": 0}},
    )
    result = run_signature_pipeline(
        FakeCC(), config, diagnosis_plots=False, start_stage="sign3"
    )
    (_, sign2, triplet_signature), kwargs = result["sign3"].fit_calls[0]
    assert triplet_signature.cctype == "sign0"
    assert sign2 is result["sign2"]
    assert kwargs["triplets_sampler"] == [FakeSampler, None, {"seed": 0}]


def test_default_triplets_use_sign1(tmp_path):
    result = run_signature_pipeline(
        FakeCC(), dataset_config(tmp_path), diagnosis_plots=False, start_stage="sign3"
    )
    (_, _, triplet_signature), kwargs = result["sign3"].fit_calls[0]
    assert triplet_signature is result["sign1"]
    assert "triplets_sampler" not in kwargs


def test_sampler_missing_from_chemicalchecker_is_reported(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "chemicalchecker.util.splitter", SimpleNamespace(__file__="old")
    )
    with pytest.raises(ImportError, match="has no BinJaccardTripletSampler"):
        resolve_triplet_sampler(TripletSamplerConfig(method="bin_jaccard"))


def test_holdout_keys_file_reaches_sign0(tmp_path):
    (tmp_path / "holdout.txt").write_text("AAAA-X\n")
    matrix_file = tmp_path / "raw.csv"
    config = dataset_config(tmp_path, holdout_keys=str(tmp_path / "holdout.txt"))
    matrix_file.write_text("inchikey,f1\nAAAA-X,1\nBBBB-X,1\n")
    result = run_signature_pipeline(
        FakeCC(), config, diagnosis_plots=False, max_stage="sign0"
    )
    _, sign0_kwargs = result["sign0"].fit_calls[0]
    assert sign0_kwargs["keys"] == ["BBBB-X"]


def test_failed_diagnosis_plot_does_not_stop_the_pipeline(tmp_path, caplog):
    result = run_signature_pipeline(
        FakeCC(), dataset_config(tmp_path), diagnosis_plots=True, max_stage="sign1"
    )
    assert list(result) == ["sign0", "sign1", "neig1"]
    assert "sign0] fitted and saved, but its diagnosis plots failed" in caplog.text
    assert "diagnosis plots failed for ['sign0', 'sign1']" in caplog.text


def test_raw_dep_profiles_reach_sign3_without_held_out_molecules(
    tmp_path, splitter_module
):
    (tmp_path / "holdout.txt").write_text("CCCC-X\n")
    config = dataset_config(
        tmp_path,
        holdout_keys=str(tmp_path / "holdout.txt"),
        triplet_sampler={
            "method": "bin_jaccard",
            "triplet_signature": "raw",
            "binarize": {"log2fc": 1.0},
        },
    )
    (tmp_path / "raw.csv").write_text("inchikey,p1\nAAAA-X,2\nBBBB-X,-3\nCCCC-X,5\n")
    result = run_signature_pipeline(
        FakeCC(), config, diagnosis_plots=False, start_stage="sign3"
    )
    (_, _, triplet_signature), _ = result["sign3"].fit_calls[0]
    assert triplet_signature.shape == (2, 2)
    assert triplet_signature.as_dataframe().to_dict("index") == {
        "AAAA-X": {"p1_up": 1, "p1_down": 0},
        "BBBB-X": {"p1_up": 0, "p1_down": 1},
    }
