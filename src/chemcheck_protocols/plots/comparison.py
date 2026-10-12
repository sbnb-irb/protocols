"""Figures of ``compare-spaces`` results, drawn from its tables only (no signatures needed).

Each plot function takes the ``comparisons`` (and ``curves``) tables written by
:func:`chemcheck_protocols.space_comparison.write_comparison` and draws whatever rows suit it,
so the same code serves any signature type; a plot with no matching rows is skipped with a log
line. Points show the mean over repetitions with its standard deviation, and the dotted line at
0.5 is chance.
"""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PIPELINE_STAGES
from . import plot_style

logger = logging.getLogger(__name__)

AUROC = "recapitulation_auroc"
SPACES_PER_FIGURE = 4  # the palette has four colours
POINT = {
    "marker": "o",
    "linestyle": "none",
    "markeredgecolor": "black",
    "markeredgewidth": 0.4,
    "elinewidth": 1.0,
    "capsize": 0,
}


def read_tables(prefixes: Sequence[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and concatenate the tables of one or more ``compare-spaces`` outputs.

    Raises
    ------
    FileNotFoundError
        If ``<prefix>_comparisons.csv`` or ``<prefix>_curves.csv`` is missing.
    """
    comparisons, curves = [], []
    for prefix in map(Path, prefixes):
        for name, frames in (("comparisons", comparisons), ("curves", curves)):
            path = prefix.parent / f"{prefix.name}_{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run compare-spaces first")
            frames.append(pd.read_csv(path))
    return (
        pd.concat(comparisons, ignore_index=True).drop_duplicates(),
        pd.concat(curves, ignore_index=True).drop_duplicates(),
    )


def slug(text: str) -> str:
    """File-name-safe version of a label."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _within_space(table: pd.DataFrame) -> pd.Series:
    return (table.a_space == table.b_space) & (table.a_signature != table.b_signature)


def _auroc_rows(table: pd.DataFrame, p_value: float) -> pd.DataFrame:
    rows = table[(table.metric == AUROC) & np.isclose(table.p_value, p_value)]
    return rows


def _stage_order(pair: tuple[str, str]) -> tuple[int, int]:
    return PIPELINE_STAGES.index(pair[0]), PIPELINE_STAGES.index(pair[1])


def _point(ax, x, y, spread, colour, horizontal=False):
    error = {"xerr": spread} if horizontal else {"yerr": spread}
    ax.errorbar(x, y, color=colour, ecolor=colour, markersize=7, **error, **POINT)


def _legend(fig, entries: list[tuple[str, str]]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=colour,
            marker="o",
            linestyle="none",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=label,
        )
        for label, colour in entries
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncols=min(len(entries), 3),
        columnspacing=1.2,
        handletextpad=0.3,
    )


def plot_recovery(
    comparisons: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
    p_value: float = 0.01,
) -> list[Path]:
    """
    Within-space recovery: for each pair of signature types of a space (x), the AUROC with which
    the higher type recovers the neighbours of the lower one (a stem from chance to the AUROC).
    Spaces side by side in colour, at most four per figure.
    """
    import matplotlib.pyplot as plt

    rows = _auroc_rows(comparisons[_within_space(comparisons)], p_value)
    rows = rows[rows.direction == "a_neighbours_by_b"]
    if rows.empty:
        logger.info("recovery: no within-space comparisons at p=%g; skipped", p_value)
        return []
    pairs = sorted(
        {(r.a_signature, r.b_signature) for r in rows.itertuples()}, key=_stage_order
    )
    spaces = list(dict.fromkeys(rows.a_space))
    written = []
    for number, start in enumerate(range(0, len(spaces), SPACES_PER_FIGURE), start=1):
        group = spaces[start : start + SPACES_PER_FIGURE]
        offsets = np.linspace(-0.22, 0.22, len(group)) if len(group) > 1 else [0.0]
        fig, ax = plt.subplots()
        for space, offset, colour in zip(group, offsets, plot_style.CATEGORICAL):
            for position, pair in enumerate(pairs):
                match = rows[
                    (rows.a_space == space)
                    & (rows.a_signature == pair[0])
                    & (rows.b_signature == pair[1])
                ]
                if match.empty:
                    continue
                row = match.iloc[0]
                ax.vlines(
                    position + offset, 0.5, row.value, color=colour, linewidth=1.2
                )
                _point(ax, position + offset, row.value, row["std"], colour)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set(
            xticks=range(len(pairs)),
            xticklabels=[f"{a}→{b}" for a, b in pairs],
            ylim=(0.45, 1.02),
            ylabel=f"Neighbour recovery AUROC (p = {p_value:g})",
        )
        _legend(fig, list(zip(group, plot_style.CATEGORICAL)))
        suffix = "" if len(spaces) <= SPACES_PER_FIGURE else f"_{number}"
        written += plot_style.save_figure(fig, output_dir / f"recovery{suffix}")
        plt.close(fig)
    return written


def plot_roc(
    comparisons: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
    p_value: float = 0.01,
) -> list[Path]:
    """
    One figure per comparison between different spaces: the ROC curve (mean, shaded one standard
    deviation) of each direction, one line per p-value cutoff, with the AUROC in the legend.
    """
    import matplotlib.pyplot as plt

    cross = comparisons[(comparisons.metric == AUROC) & ~_within_space(comparisons)]
    cross = cross[cross.a_space != cross.b_space]
    written = []
    for label in dict.fromkeys(cross.comparison):
        shown = cross[cross.comparison == label]
        names = {
            "a_neighbours_by_b": label.split(" vs ")[0],
            "b_neighbours_by_a": label.split(" vs ")[1],
        }
        other = {
            "a_neighbours_by_b": names["b_neighbours_by_a"],
            "b_neighbours_by_a": names["a_neighbours_by_b"],
        }
        fig, axes = plt.subplots(1, 2, sharey=True)
        for ax, direction in zip(axes, names):
            for colour, p in zip(
                plot_style.CATEGORICAL, sorted(shown.p_value.unique(), reverse=True)
            ):
                curve = curves[
                    (curves.comparison == label)
                    & (curves.direction == direction)
                    & np.isclose(curves.p_value, p)
                ]
                row = shown[
                    (shown.direction == direction) & np.isclose(shown.p_value, p)
                ]
                if curve.empty or row.empty:
                    continue
                auroc = row.iloc[0]
                ax.plot(
                    curve.fpr,
                    curve.tpr_mean,
                    color=colour,
                    label=f"p = {p:g}: {auroc.value:.2f} ± {auroc['std']:.2f}",
                )
                ax.fill_between(
                    curve.fpr,
                    curve.tpr_mean - curve.tpr_std,
                    curve.tpr_mean + curve.tpr_std,
                    color=colour,
                    alpha=0.3,
                    linewidth=0,
                )
            ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8)
            title = (
                f"{textwrap.fill(names[direction], 28)}\n"
                f"recovered by {textwrap.fill(other[direction], 28)}"
            )
            ax.set_title(title, fontsize=9)
            ax.set(xlabel="False positive rate")
            ax.legend(loc="lower right")
        axes[0].set_ylabel("True positive rate")
        written += plot_style.save_figure(fig, output_dir / f"roc_{slug(label)}")
        plt.close(fig)
    if not written:
        logger.info("roc: no comparisons between different spaces; skipped")
    return written


def plot_grid(
    comparisons: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
    p_value: float = 0.01,
) -> list[Path]:
    """
    Heatmaps of the AUROC between spaces (rows: side a, whose neighbours are recovered by side
    b, columns), one figure per pair of signature types; ``*`` marks values more than two
    standard deviations above the shuffled null.
    """
    import matplotlib.pyplot as plt

    rows = _auroc_rows(comparisons, p_value)
    rows = rows[
        (rows.direction == "a_neighbours_by_b") & (rows.a_space != rows.b_space)
    ]
    written = []
    for (a_sig, b_sig), group in rows.groupby(
        ["a_signature", "b_signature"], dropna=False
    ):
        values = group.pivot_table(index="a_space", columns="b_space", values="value")
        above = group.assign(
            above=group.value - group.null_value
            > 2 * np.maximum(group["std"], group.null_std)
        ).pivot_table(
            index="a_space", columns="b_space", values="above", aggfunc="first"
        )
        limit = max(np.nanmax(np.abs(values.to_numpy() - 0.5)), 0.05)
        fig, ax = plt.subplots(
            figsize=(2.8 + 1.5 * values.shape[1], 1.4 + 0.6 * values.shape[0])
        )
        image = ax.imshow(
            values.to_numpy(),
            cmap="diverging_blue_red",
            vmin=0.5 - limit,
            vmax=0.5 + limit,
            aspect="auto",
        )
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values.iat[i, j]
                if not np.isnan(value):
                    star = "*" if above.iat[i, j] == 1 else ""
                    ax.text(
                        j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=8
                    )
        ax.set(
            xticks=range(values.shape[1]),
            xticklabels=list(values.columns),
            yticks=range(values.shape[0]),
            yticklabels=list(values.index),
        )
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(
            image, ax=ax, shrink=0.7, label=f"AUROC (p = {p_value:g})"
        ).outline.set_visible(False)
        name = "grid" if pd.isna(a_sig) else f"grid_{a_sig}_{b_sig}"
        written += plot_style.save_figure(fig, output_dir / name)
        plt.close(fig)
    if not written:
        logger.info(
            "grid: no comparisons between different spaces at p=%g; skipped", p_value
        )
    return written


def plot_against_null(
    comparisons: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
    p_value: float = 0.01,
) -> list[Path]:
    """
    Every comparison as a row: the AUROC of each direction (colour) with its standard
    deviation, and the shuffled null (gray); a point that does not leave the gray one carries
    no signal.
    """
    import matplotlib.pyplot as plt

    rows = _auroc_rows(comparisons, p_value)
    if rows.empty:
        logger.info("null: no recapitulation rows at p=%g; skipped", p_value)
        return []
    labels = list(dict.fromkeys(rows.comparison))
    fig, ax = plt.subplots(figsize=(6.4, 1.0 + 0.4 * len(labels)))
    colours = {
        "a_neighbours_by_b": plot_style.BLUE,
        "b_neighbours_by_a": plot_style.RED,
    }
    for direction, offset in (
        ("a_neighbours_by_b", -0.12),
        ("b_neighbours_by_a", 0.12),
    ):
        for position, label in enumerate(labels):
            match = rows[(rows.comparison == label) & (rows.direction == direction)]
            if match.empty:
                continue
            row = match.iloc[0]
            _point(
                ax,
                row.value,
                position + offset,
                row["std"],
                colours[direction],
                horizontal=True,
            )
            ax.plot(
                row.null_value,
                position + offset,
                marker="|",
                color=plot_style.CONTEXT_GRAY,
                markersize=9,
                markeredgewidth=1.5,
                linestyle="none",
            )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set(
        yticks=range(len(labels)), yticklabels=labels, xlabel=f"AUROC (p = {p_value:g})"
    )
    ax.invert_yaxis()
    _legend(
        fig,
        [
            ("a recovered by b", plot_style.BLUE),
            ("b recovered by a", plot_style.RED),
            ("shuffled null", plot_style.CONTEXT_GRAY),
        ],
    )
    written = plot_style.save_figure(fig, output_dir / "against_null")
    plt.close(fig)
    return written


PLOTS: dict[str, Callable[..., list[Path]]] = {
    "recovery": plot_recovery,
    "roc": plot_roc,
    "grid": plot_grid,
    "against-null": plot_against_null,
}


def make_plots(
    prefixes: Sequence[str | Path],
    output_dir: str | Path,
    plots: Sequence[str] | None = None,
    p_value: float = 0.01,
    context: str = "slides",
) -> list[Path]:
    """
    Draw the chosen plots (default: all) from the tables of ``compare-spaces`` outputs.

    Raises
    ------
    ValueError
        If a plot name is unknown.
    FileNotFoundError
        If a table is missing.
    """
    unknown = sorted(set(plots or []) - set(PLOTS))
    if unknown:
        raise ValueError(f"Unknown plot(s) {unknown}; choose from {sorted(PLOTS)}")
    comparisons, curves = read_tables(prefixes)
    plot_style.apply_matplotlib(context)
    written = []
    for name in plots or PLOTS:
        written += PLOTS[name](comparisons, curves, Path(output_dir), p_value)
    logger.info("Drew %d figure file(s) in %s", len(written), output_dir)
    return written
