import numpy as np
import pandas as pd
import pytest

from chemcheck_protocols.config import BinarizationConfig, DatasetConfig
from chemcheck_protocols.triplet_profiles import (
    ProfileSignature,
    binarize_profiles,
    build_triplet_signature,
)


def test_dep_rule_calls_fold_change_or_zscore_with_direction():
    # p1: fold-change calls; p2: one clear outlier among small values (a z-score call)
    values = pd.DataFrame(
        {
            "p1": [1.2, -1.5, 0.1] + [0.0] * 27,
            "p2": [0.9] + [0.01, -0.01] * 14 + [np.nan],
        },
        index=[f"M{i:02d}" for i in range(30)],
    )
    profiles = binarize_profiles(values, BinarizationConfig(log2fc=1.0, zscore=5.0))
    # 0.9 passes p2 only through the z-score rule
    assert profiles.loc["M00", ["p1_up", "p2_up"]].tolist() == [1, 1]
    assert profiles.loc["M01", "p1_down"] == 1
    assert profiles.loc["M02"].sum() == 0
    assert profiles.loc["M29"].sum() == 0  # missing value never called


def test_fold_change_alone_ignores_zscore():
    values = pd.DataFrame(
        {"p1": [0.9] + [0.0] * 29}, index=[f"M{i}" for i in range(30)]
    )
    assert (
        binarize_profiles(values, BinarizationConfig(log2fc=1.0)).to_numpy().sum() == 0
    )


def test_percentile_rule_uses_both_sides():
    values = pd.DataFrame(
        {"c1": np.linspace(-1, 1, 21)}, index=[f"M{i}" for i in range(21)]
    )
    profiles = binarize_profiles(values, BinarizationConfig(percentile=90))
    assert profiles["c1_up"].sum() == 2 and profiles["c1_down"].sum() == 2


def test_percentile_below_zero_cutoff_is_rejected():
    values = pd.DataFrame({"c1": np.linspace(-1, 1, 21)})
    with pytest.raises(ValueError, match="higher percentile"):
        binarize_profiles(values, BinarizationConfig(percentile=40))


def test_profile_signature_looks_like_a_signature():
    profiles = pd.DataFrame({"f_up": [1, 0]}, index=["A", "B"])
    signature = ProfileSignature(profiles, "test profiles")
    assert signature.shape == (2, 1)
    assert list(signature.keys) == ["A", "B"]
    assert signature.as_dataframe() is profiles
    assert signature.data_path == "test profiles"


class FakeSign1:
    data_path = "/cc/sign1.h5"

    def as_dataframe(self):
        return pd.DataFrame(
            {"c1": [2.0, -2.0, 0.0] + [0.1] * 17}, index=[f"M{i}" for i in range(20)]
        )


def dataset_config(tmp_path, sampler):
    (tmp_path / "raw.csv").write_text("inchikey,p1\nAAAA-X,2\nAAAA-X,0\nBBBB-X,-3\n")
    return DatasetConfig.model_validate(
        {
            "key": "d6",
            "name": "D6",
            "dataset_code": "D6.002",
            "source": {"format": "wide_matrix", "path": str(tmp_path / "raw.csv")},
            "triplet_sampler": {"method": "bin_jaccard", **sampler},
        }
    )


def test_raw_profiles_average_duplicates_and_drop_held_out(tmp_path):
    config = dataset_config(
        tmp_path, {"triplet_signature": "raw", "binarize": {"log2fc": 1.0}}
    )
    signature = build_triplet_signature(None, config, {}, holdout_keys=["BBBB-X"])
    # AAAA-X appears twice (2 and 0), averaged to 1 as chemicalchecker's sign0 does
    assert signature.as_dataframe().to_dict("index") == {
        "AAAA-X": {"p1_up": 1, "p1_down": 0}
    }


def test_signature_without_binarize_is_used_as_is(tmp_path):
    sign1 = FakeSign1()
    config = dataset_config(tmp_path, {"triplet_signature": "sign1"})
    assert build_triplet_signature(None, config, {"sign1": sign1}) is sign1


def test_fitted_signature_is_binarized_by_percentile(tmp_path):
    config = dataset_config(
        tmp_path, {"triplet_signature": "sign1", "binarize": {"percentile": 90}}
    )
    signature = build_triplet_signature(None, config, {"sign1": FakeSign1()})
    frame = signature.as_dataframe()
    assert (frame.loc["M0", "c1_up"], frame.loc["M1", "c1_down"]) == (1, 1)
    assert signature.data_path == "binarized sign1 /cc/sign1.h5"
