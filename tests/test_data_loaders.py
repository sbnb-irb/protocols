import json

import numpy as np
import pytest

from chem_checker_protocols.config import DataSource
from chem_checker_protocols.data_loaders import (
    build_sign0_inputs,
    load_inchikey_mapping,
    load_wide_matrix,
)


def test_wide_matrix_is_indexed_by_inchikey(tmp_path):
    matrix_file = tmp_path / "raw.tsv"
    matrix_file.write_text("inchikey\tP1\tP2\nAAAA-X\t0.5\t1.0\nBBBB-X\t-0.2\t0.0\n")
    source = DataSource(format="wide_matrix", path=matrix_file, separator="\t")

    inputs = build_sign0_inputs(source)

    assert inputs["keys"] == ["AAAA-X", "BBBB-X"]
    assert inputs["features"] == ["P1", "P2"]
    np.testing.assert_array_equal(inputs["X"], [[0.5, 1.0], [-0.2, 0.0]])


def test_wide_matrix_without_features_is_rejected(tmp_path):
    matrix_file = tmp_path / "raw.csv"
    matrix_file.write_text("inchikey\nAAAA-X\n")
    with pytest.raises(
        ValueError, match="expected compounds as rows and features as columns"
    ):
        load_wide_matrix(DataSource(format="wide_matrix", path=matrix_file))


@pytest.mark.parametrize(
    ("data_format", "argument"), [("long_pairs", "pairs"), ("cc_h5", "data_file")]
)
def test_h5_sources_are_passed_to_chemicalchecker_as_paths(
    tmp_path, data_format, argument
):
    h5_file = tmp_path / "input.h5"
    h5_file.touch()
    assert build_sign0_inputs(DataSource(format=data_format, path=h5_file)) == {
        argument: str(h5_file)
    }


@pytest.mark.parametrize("data_format", ["wide_matrix", "long_pairs", "cc_h5"])
def test_missing_source_file_raises(tmp_path, data_format):
    with pytest.raises(FileNotFoundError):
        build_sign0_inputs(DataSource(format=data_format, path=tmp_path / "absent"))


def test_inchikey_mapping_loads_or_is_skipped(tmp_path):
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"AAAA-X": "InChI=1S/C"}))
    assert load_inchikey_mapping(mapping_file) == {"AAAA-X": "InChI=1S/C"}
    assert load_inchikey_mapping(None) is None
    with pytest.raises(FileNotFoundError):
        load_inchikey_mapping(tmp_path / "absent.json")
