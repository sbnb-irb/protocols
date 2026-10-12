import numpy as np
import pandas as pd
import pytest

from chemcheck_protocols.config import EvaluationConfig
from chemcheck_protocols.heldout_evaluation import run_evaluation, write_evaluation

N = 120
KEYS = [f"KEY{i:03d}" for i in range(N)]


class FakeSignature:
    def __init__(self, keys, vectors):
        self.keys = list(keys)
        self._rows = dict(zip(keys, vectors))

    def get_vectors(self, keys):
        return keys, np.array([self._rows[k] for k in keys])


@pytest.fixture
def space():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(N, 8))
    signatures = {
        ("full", "D6.002", "sign0"): FakeSignature(KEYS, truth),
        ("good", "D6.002", "sign3"): FakeSignature(
            KEYS, truth + rng.normal(scale=0.2, size=truth.shape)
        ),
        ("bad", "D6.002", "sign3"): FakeSignature(
            KEYS[:-5], rng.normal(size=(N - 5, 8))
        ),
        ("fold1", "D6.002", "sign3"): FakeSignature(
            KEYS, truth + rng.normal(scale=0.2, size=truth.shape)
        ),
        ("fold2", "D6.002", "sign3"): FakeSignature(
            KEYS, truth + rng.normal(scale=0.2, size=truth.shape)
        ),
    }
    return truth, lambda root, code, cctype: signatures[(root.name, code, cctype)]


def _config(tmp_path, **changes):
    (tmp_path / "held.txt").write_text("\n".join(KEYS[:30]) + "\n")
    values = {
        "reference_cc": tmp_path / "full",
        "dataset_code": "D6.002",
        "holdout": tmp_path / "held.txt",
        "runs": [
            {"name": "bad", "cc_root": tmp_path / "bad"},
            {"name": "good", "cc_root": tmp_path / "good"},
        ],
        "references": [{"name": "sign0", "signature": "sign0"}],
        "n_bootstrap": 20,
        "output": tmp_path / "out" / "eval",
        "per_molecule": True,
    }
    values.update(changes)
    return EvaluationConfig.model_validate(values)


def test_runs_are_scored_on_shared_molecules_and_resamples(tmp_path, space):
    _, opener = space
    tables = run_evaluation(_config(tmp_path), opener)
    scores = tables["scores"].set_index("run")
    assert scores.loc["good", "auroc"] > 0.9 > scores.loc["bad", "auroc"]
    assert (scores["n_molecules"] == N - 5).all()  # only molecules every run predicts
    (comparison,) = tables["comparisons"].itertuples()
    assert (comparison.first, comparison.second) == (
        "good",
        "bad",
    ) and comparison.ci_low > 0
    assert set(tables["per_molecule"].columns) == {
        "reference",
        "inchikey",
        "n_neighbours",
        "bad",
        "good",
    }
    paths = write_evaluation(tables, tmp_path / "out" / "eval")
    assert sorted(p.name for p in paths) == [
        "eval_comparisons.csv",
        "eval_per_molecule.csv",
        "eval_scores.csv",
    ]


def test_fold_models_score_their_own_held_out_molecules(tmp_path, space):
    _, opener = space
    (tmp_path / "f1.txt").write_text("\n".join(KEYS[:15]) + "\n")
    (tmp_path / "f2.txt").write_text("\n".join(KEYS[15:30]) + "\n")
    folds = [
        {"cc_root": tmp_path / "fold1", "holdout": tmp_path / "f1.txt"},
        {"cc_root": tmp_path / "fold2", "holdout": tmp_path / "f2.txt"},
    ]
    config = _config(
        tmp_path,
        holdout=None,
        per_molecule=False,
        runs=[
            {"name": "folds", "folds": folds},
            {"name": "good", "cc_root": tmp_path / "good"},
        ],
        comparisons=[["good", "folds"]],
    )
    tables = run_evaluation(config, opener)
    assert tables["scores"]["n_heldout"].tolist() == [30, 30]
    assert (
        len(tables["comparisons"]) == 2
    )  # good - folds against the baseline, then as configured


def test_binarised_profiles_can_be_the_reference(tmp_path, space):
    truth, opener = space
    pd.DataFrame(truth, index=KEYS).to_csv(tmp_path / "raw.csv")
    reference = {
        "name": "calls",
        "profiles": {"format": "wide_matrix", "path": tmp_path / "raw.csv"},
        "binarize": {"log2fc": 1.0},
    }
    tables = run_evaluation(
        _config(tmp_path, references=[reference], per_molecule=False), opener
    )
    assert tables["scores"]["reference"].unique().tolist() == ["calls"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"comparisons": [["good", "missing"]]}, "Unknown run name"),
        ({"holdout": None}, "holdout is required"),
        ({"references": [{"name": "x"}]}, "either signature or profiles"),
        (
            {"runs": [{"name": "a", "cc_root": "x"}, {"name": "a", "cc_root": "y"}]},
            "Duplicate run",
        ),
    ],
)
def test_inconsistent_evaluations_are_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **changes)
