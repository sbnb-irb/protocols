import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from chemcheck_protocols.config import ComparisonConfig
from chemcheck_protocols.space_comparison import (
    SpaceStore,
    run_comparison,
    shared_vectors,
    write_comparison,
)

N = 200
KEYS = [f"KEY{i:03d}" for i in range(N)]
UNIVERSE = KEYS + [
    f"EXTRA{i:03d}" for i in range(100)
]  # sign3 also covers unmeasured compounds


class FakeSignature:
    def __init__(self, keys, vectors):
        self.keys = list(keys)
        self._rows = dict(zip(keys, vectors))

    def get_vectors(self, keys):
        found = [k for k in keys if k in self._rows]
        return found, np.array([self._rows[k] for k in found])


@pytest.fixture
def opener():
    """Spaces ``a`` and ``b`` share structure; ``c`` is unrelated and lacks 20 compounds."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(N, 8))
    related = truth + rng.normal(scale=0.3, size=truth.shape)
    unrelated = rng.normal(size=(N - 20, 8))
    signatures = {}
    for name, keys, vectors in (
        ("a", KEYS, truth),
        ("b", KEYS, related),
        ("c", KEYS[20:], unrelated),
    ):
        for cctype in ("sign0", "sign1", "sign2"):
            signatures[(name, cctype)] = FakeSignature(keys, vectors)
        extra = rng.normal(size=(len(UNIVERSE), 8))
        signatures[(name, "sign3")] = FakeSignature(UNIVERSE, extra)
    return lambda root, code, cctype: signatures[(root.name, cctype)]


def make_config(tmp_path, **changes):
    values = {
        "spaces": [
            {"name": name, "cc_root": tmp_path / name, "dataset_code": "D1.003"}
            for name in "abc"
        ],
        "pairs": [["a", "b"], ["a", "c"]],
        "n_permutations": 5,
        "output": tmp_path / "out" / "compare",
    }
    values.update(changes)
    return ComparisonConfig.model_validate(values)


def metric(table, metric_name, **where):
    rows = table[table.metric == metric_name]
    for column, value in where.items():
        rows = rows[rows[column] == value]
    return rows


def test_a_related_pair_beats_its_shuffled_null_and_an_unrelated_pair_does_not(
    opener, tmp_path
):
    table = run_comparison(make_config(tmp_path), opener)["comparisons"]
    related = metric(table, "recapitulation_auroc", b_space="b", p_value=0.01).iloc[0]
    unrelated = metric(table, "recapitulation_auroc", b_space="c", p_value=0.01).iloc[0]
    assert related.value > 0.8
    assert related.null_value == pytest.approx(0.5, abs=0.05)
    assert unrelated.value == pytest.approx(unrelated.null_value, abs=0.1)
    assert metric(table, "distance_spearman", b_space="b").iloc[0].value > 0.5


def test_each_pair_is_scored_in_both_directions_at_every_cutoff_with_its_curve(
    opener, tmp_path
):
    tables = run_comparison(make_config(tmp_path, pairs=[["a", "b"]]), opener)
    auroc = metric(tables["comparisons"], "recapitulation_auroc")
    assert sorted(auroc.direction.unique()) == [
        "a_neighbours_by_b",
        "b_neighbours_by_a",
    ]
    assert sorted(auroc.p_value.unique()) == [0.001, 0.01]
    curves = tables["curves"]
    assert len(curves) == 4 * 101
    assert curves.fpr.min() == 0 and curves.fpr.max() == 1


def test_only_compounds_in_both_spaces_are_compared(opener, tmp_path):
    table = run_comparison(make_config(tmp_path, pairs=[["a", "c"]]), opener)[
        "comparisons"
    ]
    assert set(table.n_compounds) == {N - 20}


def test_measured_compounds_exclude_the_predicted_ones_at_sign3(opener, tmp_path):
    base = {"pairs": [["a", "b"]], "signatures": ["sign3"]}
    measured = run_comparison(make_config(tmp_path, **base), opener)["comparisons"]
    assert set(measured.n_compounds) == {N}
    every = make_config(tmp_path, compounds="all_shared", **base)
    assert set(run_comparison(every, opener)["comparisons"].n_compounds) == {
        len(UNIVERSE)
    }
    drawn = make_config(tmp_path, compounds="all_shared", max_compounds=120, **base)
    assert set(run_comparison(drawn, opener)["comparisons"].n_compounds) == {120}


def test_compounds_can_come_from_a_file(opener, tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("\n".join(KEYS[:80] + ["NOT_THERE"]))
    config = make_config(
        tmp_path, compounds="from_file", compounds_file=keys, pairs=[["a", "b"]]
    )
    assert set(run_comparison(config, opener)["comparisons"].n_compounds) == {80}


def test_a_matrix_file_can_be_a_side_and_missing_values_count_as_zero(opener, tmp_path):
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(
        rng.normal(size=(N, 6)),
        index=pd.Index(KEYS, name="inchikey"),
        columns=list("pqrstu"),
    )
    matrix.iloc[::7, 0] = np.nan
    matrix.to_csv(tmp_path / "matrix.csv")
    config = make_config(
        tmp_path,
        spaces=[
            {"name": "a", "cc_root": tmp_path / "a", "dataset_code": "D1.003"},
            {"name": "raw", "matrix": tmp_path / "matrix.csv"},
        ],
        pairs=[["raw", "a"]],
    )
    assert [c.label for c in config.comparisons()] == ["raw vs a@sign2"]
    table = run_comparison(config, opener)["comparisons"]
    assert set(table.n_compounds) == {N}


def test_shorthands_expand_into_one_list_without_repeats(tmp_path):
    config = make_config(
        tmp_path,
        signatures=["sign1", "sign2"],
        pairs=[["a", "b"]],
        against=[{"reference": "c", "spaces": ["a", "b"]}],
        within_space=[["sign0", "sign2"]],
        comparisons=[
            {
                "a": {"space": "a", "signature": "sign1"},
                "b": {"space": "b", "signature": "sign1"},
            },
            {
                "a": {"space": "a", "signature": "sign0"},
                "b": {"space": "b", "signature": "sign3"},
            },
        ],
    )
    labels = [c.label for c in config.comparisons()]
    assert len(labels) == len(set(labels)) == 2 + 4 + 3 + 1
    assert "a@sign1 vs b@sign1" in labels and "c@sign0 vs c@sign2" in labels
    assert "a@sign0 vs b@sign3" in labels


def test_within_space_recovery_goes_from_the_lower_to_the_higher_stage(
    opener, tmp_path
):
    config = make_config(tmp_path, pairs=[], within_space=[["sign0", "sign1"]])
    table = run_comparison(config, opener)["comparisons"]
    assert set(table.comparison) == {
        "a@sign0 vs a@sign1",
        "b@sign0 vs b@sign1",
        "c@sign0 vs c@sign1",
    }
    assert metric(
        table, "recapitulation_auroc", a_space="a", p_value=0.01
    ).value.min() == pytest.approx(1.0)


def test_zero_vectors_are_dropped_and_too_few_compounds_are_refused(tmp_path):
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(N, 4))
    zero_row = vectors.copy()
    zero_row[0] = 0
    signatures = {
        ("a", "sign2"): FakeSignature(KEYS, zero_row),
        ("b", "sign2"): FakeSignature(KEYS, vectors),
    }
    config = make_config(tmp_path, pairs=[["a", "b"]])
    store = SpaceStore(
        config, lambda root, code, cctype: signatures[(root.name, cctype)]
    )
    comparison = config.comparisons()[0]
    keys, kept_a, kept_b = shared_vectors(store, comparison, KEYS)
    assert len(keys) == N - 1 and "KEY000" not in keys and len(kept_a) == len(kept_b)
    with pytest.raises(ValueError, match="at least 50"):
        shared_vectors(store, comparison, KEYS[:30])


def test_results_are_written_with_a_record_of_the_run(opener, tmp_path):
    config = make_config(tmp_path, pairs=[["a", "b"]])
    paths = write_comparison(run_comparison(config, opener), config)
    assert sorted(p.name for p in paths) == [
        "compare_comparisons.csv",
        "compare_curves.csv",
        "compare_run.yaml",
    ]
    record = yaml.safe_load(paths[-1].read_text())
    assert record["compounds"] == "measured"
    assert record["comparisons"] == ["a@sign2 vs b@sign2"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"pairs": [["a", "zzz"]]}, "Unknown space"),
        ({"pairs": [], "within_space": []}, "Nothing to compare"),
        ({"within_space": [["sign2", "sign1"]]}, "higher stage"),
        ({"compounds": "from_file"}, "compounds_file"),
        ({"p_value_cutoffs": [0.01, 2]}, "between 0 and 1"),
        (
            {
                "comparisons": [
                    {"a": {"space": "a"}, "b": {"space": "b", "signature": "sign2"}}
                ]
            },
            "give a signature",
        ),
    ],
)
def test_inconsistent_comparisons_are_rejected(tmp_path, changes, message):
    with pytest.raises(ValidationError, match=message):
        make_config(tmp_path, **changes)


def test_a_pair_of_one_space_with_itself_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="itself"):
        make_config(tmp_path, pairs=[["a", "a"]])


def test_the_null_is_the_same_with_or_without_precomputed_distances():
    from scipy.spatial.distance import pdist, squareform

    from chemcheck_protocols.evaluation import background_distance_cutoff
    from chemcheck_protocols.space_comparison import shuffled_null

    rng = np.random.default_rng(5)
    reference = rng.normal(size=(150, 6))
    query = reference + rng.normal(scale=0.3, size=reference.shape)
    cutoff, _ = background_distance_cutoff(reference, 0.05, random_state=0)
    distances = (
        squareform(pdist(reference, "cosine")),
        squareform(pdist(query, "cosine")),
    )
    fast = shuffled_null(
        reference, query, 30, 120, 0.05, 0, cutoff=cutoff, distances=distances
    )
    slow = shuffled_null(reference, query, 30, 120, 0.05, 0)
    assert fast[0] == pytest.approx(0.5, abs=0.03) and slow[0] == pytest.approx(
        0.5, abs=0.03
    )
    assert fast[1] == pytest.approx(slow[1], abs=0.02)


def test_measured_compounds_are_those_kept_through_sign0_to_sign2(opener, tmp_path):
    """The sanitizer can drop input compounds from sign1 and sign2; sign0 and sign3 still hold them."""
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(N, 8))
    signatures = {
        ("d", "sign0"): FakeSignature(KEYS, vectors),
        ("d", "sign1"): FakeSignature(KEYS[30:], vectors[30:]),
        ("d", "sign2"): FakeSignature(KEYS[30:], vectors[30:]),
        ("d", "sign3"): FakeSignature(KEYS, vectors),
    }
    config = make_config(
        tmp_path,
        spaces=[{"name": "d", "cc_root": tmp_path / "d", "dataset_code": "D1.003"}],
        pairs=[],
        within_space=[["sign0", "sign1"], ["sign0", "sign3"]],
    )
    table = run_comparison(
        config, lambda root, code, cctype: signatures[(root.name, cctype)]
    )
    assert set(table["comparisons"].n_compounds) == {N - 30}
