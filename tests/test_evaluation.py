import numpy as np
import pytest

from chem_checker_protocols.evaluation import (
    cosine_nn_recapitulation_auroc,
    get_shared_vectors,
    shared_key_recapitulation,
)


@pytest.fixture
def vectors():
    return np.random.default_rng(0).normal(size=(400, 16))


def test_identical_spaces_recapitulate_perfectly(vectors):
    result = cosine_nn_recapitulation_auroc(
        vectors, vectors, n_random=200, n_subsamples=3, random_state=1
    )
    assert result["auroc"] == pytest.approx(1.0)


def test_unrelated_spaces_recapitulate_at_chance(vectors):
    unrelated = np.random.default_rng(1).normal(size=vectors.shape)
    result = cosine_nn_recapitulation_auroc(
        vectors, unrelated, n_random=400, n_subsamples=3, random_state=1
    )
    assert result["auroc"] == pytest.approx(0.5, abs=0.1)


def test_recapitulation_is_reproducible_with_a_seed(vectors):
    noisy = vectors + np.random.default_rng(2).normal(scale=0.5, size=vectors.shape)
    first = cosine_nn_recapitulation_auroc(
        vectors, noisy, n_random=200, n_subsamples=3, random_state=7
    )
    second = cosine_nn_recapitulation_auroc(
        vectors, noisy, n_random=200, n_subsamples=3, random_state=7
    )
    assert first == second


def test_shared_key_recapitulation_runs_both_directions(vectors):
    result = shared_key_recapitulation(
        vectors, vectors, len(vectors), n_random=200, n_subsamples=2, random_state=3
    )
    assert result["a_recap_by_b"]["auroc"] == pytest.approx(1.0)
    assert result["b_recap_by_a"]["auroc"] == pytest.approx(1.0)


class FakeSignature:
    def __init__(self, keys, vectors):
        self.keys = list(keys)
        self._rows = dict(zip(keys, vectors))

    def get_vectors(self, keys):
        found = sorted(k for k in keys if k in self._rows)
        return np.array(found), np.array([self._rows[k] for k in found])


def test_shared_vectors_are_row_aligned_across_spaces(vectors):
    keys = [f"K{i:04d}" for i in range(len(vectors))]
    signature_a = FakeSignature(keys, vectors)
    signature_b = FakeSignature(
        keys[::-1], -vectors[::-1]
    )  # same molecules, other order

    shared, vectors_a, vectors_b = get_shared_vectors(
        signature_a, signature_b, max_pool=100, random_state=4
    )

    assert len(shared) == 100
    np.testing.assert_array_equal(vectors_a, -vectors_b)


def test_too_few_shared_molecules_is_rejected(vectors):
    with pytest.raises(ValueError, match="compounds shared"):
        get_shared_vectors(
            FakeSignature(["A"], vectors[:1]), FakeSignature(["A"], vectors[:1])
        )
