import pytest

from chemcheck_protocols.data_loaders import load_key_list
from chemcheck_protocols.holdouts import (
    disjoint_folds,
    random_holdout,
    stratified_folds,
    write_key_list,
)

KEYS = [f"KEY{i:03d}" for i in range(50)]


def test_random_holdout_is_reproducible_and_ignores_key_order():
    first = random_holdout(KEYS, 0.2, seed=0)
    assert first == random_holdout(list(reversed(KEYS)), 0.2, seed=0)
    assert len(first) == 10 and first == sorted(first) and set(first) <= set(KEYS)
    assert first != random_holdout(KEYS, 0.2, seed=1)


@pytest.mark.parametrize("fraction", [0, 1, 0.001])
def test_random_holdout_rejects_empty_or_full_fractions(fraction):
    with pytest.raises(ValueError):
        random_holdout(KEYS, fraction, seed=0)


def test_disjoint_folds_cover_the_pool_once():
    folds = disjoint_folds(KEYS, 4, seed=1)
    assert [len(f) for f in folds] == [13, 13, 12, 12]
    assert sorted(k for fold in folds for k in fold) == KEYS


def test_stratified_folds_spread_every_group():
    groups = {k: ("a" if i < 30 else "b") for i, k in enumerate(KEYS)}
    folds = stratified_folds(groups, 5, seed=0)
    assert sorted(k for fold in folds for k in fold) == KEYS
    for fold in folds:
        assert sum(groups[k] == "a" for k in fold) == 6
        assert sum(groups[k] == "b" for k in fold) == 4


def test_written_key_list_reads_back(tmp_path):
    path = write_key_list(tmp_path / "out" / "held.txt", ["B", "A"], "two keys\nseed 0")
    assert load_key_list(path) == ["B", "A"]
    assert path.read_text().startswith("# two keys\n# seed 0\n")
