"""Shared typography and export settings for Figure 6 panels."""

from __future__ import annotations

import matplotlib as mpl


AXIS_LABEL_PT = 7
TICK_LABEL_PT = 6
SCALE_LABEL_PT = 6
LEGEND_LABEL_PT = 6


def apply_figure6_style() -> None:
    """Apply the manuscript typography used by current and future panels."""
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": TICK_LABEL_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_LABEL_PT,
        "ytick.labelsize": TICK_LABEL_PT,
        "legend.fontsize": LEGEND_LABEL_PT,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
