import pytest

from chemcheck_protocols.archiving import find_regenerable_files, format_size
from chemcheck_protocols.cli import main


def make_sign3_models(cc_root, code="A1.001", molecule_set="full"):
    models = cc_root / molecule_set / code[0] / code[:2] / code / "sign3" / "models"
    models.mkdir(parents=True)
    for name in (
        "all_sign2.h5",
        "all_sign2_coverage.h5",
        "all_sign2.complete.h5",
        "train.h5",
        "traintest_final.h5",
    ):
        (models / name).write_bytes(b"x" * 100)
    return models


def test_only_regenerable_files_are_listed(tmp_path):
    models = make_sign3_models(tmp_path / "cc")
    found = find_regenerable_files(tmp_path / "cc")
    assert [path.name for path in found] == [
        "all_sign2.complete.h5",
        "all_sign2.h5",
        "all_sign2_coverage.h5",
    ]
    assert all(path.parent == models for path in found)


def test_both_molecule_sets_are_listed(tmp_path):
    make_sign3_models(tmp_path / "cc")
    make_sign3_models(tmp_path / "cc", molecule_set="reference")
    found = find_regenerable_files(tmp_path / "cc")
    assert len(found) == 6
    assert {path.relative_to(tmp_path / "cc").parts[0] for path in found} == {
        "full",
        "reference",
    }


def test_linked_reference_spaces_are_never_listed(tmp_path):
    make_sign3_models(tmp_path / "release", "B1.001")
    reference = tmp_path / "release" / "full" / "B" / "B1" / "B1.001"
    cc_root = tmp_path / "cc"
    (cc_root / "full" / "B" / "B1").mkdir(parents=True)
    (cc_root / "full" / "B" / "B1" / "B1.001").symlink_to(reference)
    assert find_regenerable_files(cc_root) == []


def test_a_folder_without_full_is_not_an_instance(tmp_path):
    with pytest.raises(ValueError, match="not a CC instance"):
        find_regenerable_files(tmp_path)


def test_prune_instance_deletes_only_with_apply(tmp_path):
    models = make_sign3_models(tmp_path / "cc")
    assert main(["prune-instance", str(tmp_path / "cc")]) == 0
    assert len(list(models.glob("all_sign2*.h5"))) == 3
    assert main(["prune-instance", str(tmp_path / "cc"), "--apply"]) == 0
    assert sorted(path.name for path in models.iterdir()) == [
        "train.h5",
        "traintest_final.h5",
    ]


def test_prune_instance_rejects_a_missing_folder(tmp_path):
    assert main(["prune-instance", str(tmp_path / "nowhere")]) == 2


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (512, "512.0 B"),
        (1536, "1.5 KiB"),
        (3 * 1024**3, "3.0 GiB"),
        (2 * 1024**4, "2.0 TiB"),
    ],
)
def test_format_size(size_bytes, expected):
    assert format_size(size_bytes) == expected
