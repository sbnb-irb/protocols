import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from chemcheck_protocols.cli import main
from chemcheck_protocols.plots.comparison import make_plots, read_tables
from chemcheck_protocols.space_comparison import COMPARISON_COLUMNS, CURVE_COLUMNS

FPR = np.linspace(0, 1, 11)


def auroc_row(a, a_sig, b, b_sig, direction, p_value, value):
    return {
        "comparison": f"{a}@{a_sig} vs {b}@{b_sig}",
        "a_space": a,
        "a_signature": a_sig,
        "b_space": b,
        "b_signature": b_sig,
        "metric": "recapitulation_auroc",
        "direction": direction,
        "p_value": p_value,
        "value": value,
        "std": 0.01,
        "null_value": 0.5,
        "null_std": 0.01,
        "mean_positive_rate": 0.01,
        "n_compounds": 200,
    }


@pytest.fixture
def prefix(tmp_path):
    rows, curves = [], []
    for space in ("tx", "prot", "raw"):
        for lower, higher, value in (("sign0", "sign1", 0.95), ("sign0", "sign2", 0.8)):
            rows.append(
                auroc_row(space, lower, space, higher, "a_neighbours_by_b", 0.01, value)
            )
    for direction in ("a_neighbours_by_b", "b_neighbours_by_a"):
        for p_value in (0.01, 0.001):
            rows.append(
                auroc_row("tx", "sign2", "prot", "sign2", direction, p_value, 0.6)
            )
            curves += [
                {
                    "comparison": "tx@sign2 vs prot@sign2",
                    "direction": direction,
                    "p_value": p_value,
                    "fpr": fpr,
                    "tpr_mean": fpr**0.5,
                    "tpr_std": 0.02,
                }
                for fpr in FPR
            ]
    base = tmp_path / "results" / "run"
    base.parent.mkdir()
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(
        f"{base}_comparisons.csv", index=False
    )
    pd.DataFrame(curves, columns=CURVE_COLUMNS).to_csv(
        f"{base}_curves.csv", index=False
    )
    return base


def names(paths):
    return sorted(p.name for p in paths if p.suffix == ".png")


def test_every_plot_is_drawn_from_the_tables_alone(prefix, tmp_path):
    written = make_plots([prefix], tmp_path / "figures")
    assert names(written) == [
        "against_null.png",
        "grid_sign2_sign2.png",
        "recovery.png",
        "roc_tx_sign2_vs_prot_sign2.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)
    assert {path.parent.name for path in written} == {"svg", "png"}


def test_a_plot_with_no_matching_rows_is_skipped(prefix, tmp_path):
    only_cross = pd.read_csv(f"{prefix}_comparisons.csv")
    only_cross = only_cross[only_cross.a_space != only_cross.b_space]
    only_cross.to_csv(f"{prefix}_comparisons.csv", index=False)
    assert names(make_plots([prefix], tmp_path / "figures", ["recovery", "roc"])) == [
        "roc_tx_sign2_vs_prot_sign2.png"
    ]


def test_many_spaces_are_split_over_several_recovery_figures(prefix, tmp_path):
    table = pd.read_csv(f"{prefix}_comparisons.csv")
    extra = table[(table.a_space == "tx") & (table.b_signature == "sign1")].copy()
    for number in range(4):
        copy = extra.copy()
        copy["a_space"] = copy["b_space"] = f"more{number}"
        table = pd.concat([table, copy])
    table.to_csv(f"{prefix}_comparisons.csv", index=False)
    assert names(make_plots([prefix], tmp_path / "figures", ["recovery"])) == [
        "recovery_1.png",
        "recovery_2.png",
    ]


def test_several_runs_can_be_drawn_together(prefix, tmp_path):
    comparisons, curves = read_tables([prefix, prefix])
    assert len(comparisons) == len(pd.read_csv(f"{prefix}_comparisons.csv"))
    assert len(curves) == len(pd.read_csv(f"{prefix}_curves.csv"))


def test_unknown_plots_and_missing_tables_are_refused(prefix, tmp_path):
    with pytest.raises(ValueError, match="Unknown plot"):
        make_plots([prefix], tmp_path, ["pie"])
    with pytest.raises(FileNotFoundError, match="compare-spaces"):
        make_plots([tmp_path / "nothing"], tmp_path)


def test_the_command_draws_figures_and_reports_a_missing_table(prefix, tmp_path):
    figures = tmp_path / "figures"
    code = main(
        [
            "plot-comparison",
            "--tables",
            str(prefix),
            "--output-dir",
            str(figures),
            "--plots",
            "recovery",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert code == 0 and (figures / "png" / "recovery.png").exists()
    code = main(
        [
            "plot-comparison",
            "--tables",
            str(tmp_path / "nothing"),
            "--output-dir",
            str(figures),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert code == 2


def test_comparisons_with_dots_in_their_names_get_their_own_figure_files(
    prefix, tmp_path
):
    for name in ("comparisons", "curves"):
        table = pd.read_csv(f"{prefix}_{name}.csv")
        for column in ("comparison", "a_space", "b_space"):
            if column in table:
                table[column] = (
                    table[column]
                    .str.replace("tx", "D1.001 tx")
                    .str.replace("prot", "D6.002 prot")
                )
        table.to_csv(f"{prefix}_{name}.csv", index=False)
    written = make_plots([prefix], tmp_path / "figures", ["roc"])
    assert names(written) == ["roc_d1_001_tx_sign2_vs_d6_002_prot_sign2.png"]
