"""Plot style (matplotlib): colours, colour maps, sizes for paper or slides, and saving.

The lab's shared plot look: four palette colours, one open sans-serif font, no grid, and sizes
for paper or slides. Copied without its Plotly part so the style travels with the code. Typical use::

    from chemcheck_protocols.plots import plot_style
    plot_style.apply_matplotlib("slides")      # or "paper"
    fig, ax = plt.subplots()
    ...
    plot_style.save_figure(fig, "figures/name")  # figures/svg/name.svg and figures/png/name.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Colours by role. Groups take them in this order; map them by category name.
BLUE = "#326586"
RED = "#A8505E"
TEAL = "#2A9D8F"
GRAY_VIOLET = "#8C7FA6"
CATEGORICAL = [BLUE, RED, TEAL, GRAY_VIOLET]
CONTEXT_GRAY = "#BBBBBB"  # background data, "not significant", non-highlighted groups
FILL_ALPHA = 0.7  # filled marks (bars, histograms, areas, violins); lines stay opaque
POINT_ALPHA = 0.6  # scatter points, so overlaps stay visible
WHITE = "#FFFFFF"

# Heatmaps: blue = low, red = high. Diverging for data with a natural centre
# (log2 fold change, z-score, correlation); sequential for one-direction data.
DIVERGING = [BLUE, WHITE, RED]
SEQUENTIAL = [WHITE, BLUE]

FONT_FAMILY = ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"]

Context = Literal["paper", "slides"]

# Style sheets per output medium: the base sheet holds the look and the paper sizes; the slides
# sheet overrides the sizes. Both are plain matplotlib style files, so they can be edited as such.
_STYLE_FILES: dict[str, list[Path]] = {
    "paper": [Path(__file__).with_name("plot_style.mplstyle")],
    "slides": [
        Path(__file__).with_name("plot_style.mplstyle"),
        Path(__file__).with_name("plot_style_slides.mplstyle"),
    ],
}


def _check_context(context: str) -> None:
    if context not in _STYLE_FILES:
        raise ValueError(
            f"Unknown context {context!r}; expected one of {sorted(_STYLE_FILES)}."
        )


def apply_matplotlib(context: Context = "paper") -> None:
    """Apply the plot style to matplotlib (and seaborn) for the rest of the session.

    Registers the colour maps ``diverging_blue_red`` and ``sequential_white_blue`` and makes
    the sequential one the default for images and heatmaps.
    """
    _check_context(context)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.style.use(_STYLE_FILES[context])
    for name, colours in (
        ("diverging_blue_red", DIVERGING),
        ("sequential_white_blue", SEQUENTIAL),
    ):
        if name not in mpl.colormaps:
            cmap = LinearSegmentedColormap.from_list(name, colours)
            mpl.colormaps.register(cmap)
            mpl.colormaps.register(cmap.reversed(), name=f"{name}_r")
    mpl.rcParams["image.cmap"] = "sequential_white_blue"


def save_figure(
    fig: Any, path: str | Path, formats: tuple[str, ...] = ("svg", "png")
) -> list[Path]:
    """Save a matplotlib figure in each format; return the written paths.

    ``path`` is the folder plus the file name without extension; each format goes to its
    own subfolder (``figures/volcano`` -> ``figures/svg/volcano.svg``,
    ``figures/png/volcano.png``) so a folder of figures stays easy to browse.
    """
    stem = Path(path)
    if stem.suffix.lower() in {f".{fmt}" for fmt in formats}:
        stem = stem.with_suffix("")  # a dot inside a name (D1.001) is not an extension
    written = []
    for fmt in formats:
        target = stem.parent / fmt / f"{stem.name}.{fmt}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, format=fmt)
        written.append(target)
    logger.info("Saved figure: %s", ", ".join(str(p) for p in written))
    return written
