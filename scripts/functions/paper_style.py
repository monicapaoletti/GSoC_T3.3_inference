"""One place to set how every paper figure looks.

The problem this solves: the figures were authored 7-15 inches wide and then squeezed
into a 6.5-inch text block by \includegraphics, scaling everything down by 0.43x-0.74x.
A 10 pt tick label rendered at 4.3-7.4 pt against 11 pt body text, which is why the
labels had to be zoomed to read. Bumping font sizes alone would not fix line weights and
markers, which shrink by the same factor.

The fix is to author each figure AT its final on-page size, so the scale factor is 1 and
every dimension -- fonts, line widths, marker sizes -- renders exactly as chosen.

Usage:
    import paper_style
    paper_style.use()                          # set the rcParams
    fig, ax = plt.subplots(figsize=paper_style.figsize())          # full \linewidth
    fig, ax = plt.subplots(figsize=paper_style.figsize(0.7))       # 0.7\linewidth
    fig, axes = plt.subplots(1, 3, figsize=paper_style.figsize(1.0, panels=3))
"""
import matplotlib as mpl

# elsarticle preprint, 11pt, with the geometry package as loaded in main.tex.
TEXT_WIDTH_IN = 6.5
GOLDEN = 0.618

# Rendered 1:1, so these ARE the point sizes on the page. Body text is 11 pt and
# captions about 10 pt; axis furniture sits just below the caption so it is comfortably
# readable without competing with the prose.
SIZES = {
    "font.size":        10.5,
    "axes.labelsize":   11.0,
    "axes.titlesize":   11.5,
    "xtick.labelsize":  10.0,
    "ytick.labelsize":  10.0,
    "legend.fontsize":  10.0,
    "figure.titlesize": 12.0,
}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def figsize(width_frac=1.0, aspect=GOLDEN, panels=1):
    """Figure size in inches for a figure included at `width_frac`*\\linewidth.

    `panels` divides the aspect so a row of n panels stays readable rather than becoming
    a letterbox: each panel keeps roughly the single-panel proportions.
    """
    w = TEXT_WIDTH_IN * width_frac
    h = w * aspect / (1 if panels == 1 else panels ** 0.62)
    return (w, h)


def use(grid=True):
    """Apply the shared style. Call once, before creating figures."""
    mpl.rcParams.update(SIZES)
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8, "lines.markersize": 5.0,
        "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": grid, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.7, "axes.axisbelow": True,
    })


def grid_figsize(nrow=1, ncol=1, width_frac=1.0, panel_aspect=0.80):
    """Size for an nrow x ncol panel grid included at `width_frac`*\\linewidth.

    The TOTAL width is pinned to the on-page width, so a 2x2 grid at \\linewidth gets
    3.25in panels rather than 5.2in ones scaled down to 3.25in. Same pixels on the page
    either way; the difference is that the fonts and line weights are chosen at the size
    they will actually be seen, instead of being shrunk by \\includegraphics.
    """
    w = TEXT_WIDTH_IN * width_frac
    return (w, nrow * (w / ncol) * panel_aspect)
