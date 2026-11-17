"""Render Figure 6 semantic-decoding statistical review panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binomtest, fisher_exact


from figure6_plot_style import LEGEND_LABEL_PT, SCALE_LABEL_PT, apply_figure6_style


FIGSIZE = (1.8, 1.5)
WIDE_FIGSIZE = (3.95, 0.9)
PROBES = ("Probe 1", "Probe 2")
PHASES = ("Before retraction", "After retraction")
LEGEND_PHASES = ("Before", "After")
COLORS = ("#d55e00", "#0072b2")
COUNTS = np.zeros((2, 2), dtype=int)
TOTALS = np.zeros((2, 2), dtype=int)
CHANCE = 0.05


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return a two-sided exact binomial confidence interval."""
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def calculate_statistics() -> dict[str, object]:
    proportions = COUNTS / TOTALS
    intervals = np.empty((2, 2, 2), dtype=float)
    chance_p = np.empty((2, 2), dtype=float)
    comparison_p = np.empty(2, dtype=float)
    for probe in range(2):
        for phase in range(2):
            k = int(COUNTS[probe, phase])
            n = int(TOTALS[probe, phase])
            intervals[probe, phase] = clopper_pearson(k, n)
            chance_p[probe, phase] = binomtest(
                k, n, CHANCE, alternative="greater"
            ).pvalue
        table = np.array([
            [COUNTS[probe, 0], TOTALS[probe, 0] - COUNTS[probe, 0]],
            [COUNTS[probe, 1], TOTALS[probe, 1] - COUNTS[probe, 1]],
        ])
        comparison_p[probe] = fisher_exact(table, alternative="two-sided").pvalue
    return {
        "proportions": proportions,
        "intervals": intervals,
        "chance_p": chance_p,
        "comparison_p": comparison_p,
    }


def significance_symbol(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_bracket(ax: plt.Axes, x0: float, x1: float, y: float, label: str) -> None:
    height = 0.7
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="black", lw=0.5)
    ax.text(
        (x0 + x1) / 2,
        y + height + 0.25,
        label,
        ha="center",
        va="bottom",
        fontsize=SCALE_LABEL_PT,
    )


def style_axes(ax: plt.Axes) -> None:
    ax.axhline(
        100 * CHANCE,
        color="#777777",
        lw=0.6,
        ls=(0, (3, 2)),
        zorder=0,
    )
    ax.text(0.5, 5.45, "Chance", ha="center", va="bottom", color="#555555")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 37)
    ax.set_xticks([0, 1], PROBES)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Significant neurons (%)", labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", direction="in", pad=2)


def add_statistics(
    ax: plt.Axes,
    positions: np.ndarray,
    intervals_percent: np.ndarray,
    chance_p: np.ndarray,
    comparison_p: np.ndarray,
    *,
    comparison_label: str = "pvalue",
    compact: bool = False,
) -> None:
    for probe in range(2):
        for phase in range(2):
            ax.text(
                positions[probe, phase],
                intervals_percent[probe, phase, 1] + 0.65,
                significance_symbol(float(chance_p[probe, phase])),
                ha="center",
                va="bottom",
                fontsize=SCALE_LABEL_PT,
            )
        if compact:
            bracket_y = 36.3
        else:
            bracket_y = 31.2 if probe == 0 else 33.5
        label = (
            significance_symbol(float(comparison_p[probe]))
            if comparison_label == "symbol"
            else f"P={comparison_p[probe]:.3f}".replace("0.", ".")
        )
        add_bracket(
            ax,
            positions[probe, 0],
            positions[probe, 1],
            bracket_y,
            label,
        )


def render_panel(
    output_stem: Path,
    *,
    mode: str,
    statistics: dict[str, object],
    figsize: tuple[float, float] = FIGSIZE,
    legend_side: bool = False,
    compact: bool = False,
) -> list[Path]:
    proportions = np.asarray(statistics["proportions"]) * 100
    intervals = np.asarray(statistics["intervals"]) * 100
    chance_p = np.asarray(statistics["chance_p"])
    comparison_p = np.asarray(statistics["comparison_p"])
    centers = np.array([0.0, 1.0])
    offsets = np.array([-0.17, 0.17])
    positions = centers[:, None] + offsets[None, :]

    fig, ax = plt.subplots(figsize=figsize)
    for phase in range(2):
        values = proportions[:, phase]
        lower = values - intervals[:, phase, 0]
        upper = intervals[:, phase, 1] - values
        if mode == "points":
            ax.errorbar(
                positions[:, phase],
                values,
                yerr=np.vstack((lower, upper)),
                fmt="o",
                ms=3.4,
                mfc=COLORS[phase],
                mec=COLORS[phase],
                ecolor=COLORS[phase],
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.7,
                label=LEGEND_PHASES[phase],
                zorder=3,
            )
        elif mode == "bars":
            ax.bar(
                positions[:, phase],
                values,
                width=0.27,
                color=COLORS[phase],
                linewidth=0,
                label=LEGEND_PHASES[phase],
                zorder=2,
            )
            ax.errorbar(
                positions[:, phase],
                values,
                yerr=np.vstack((lower, upper)),
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
                capsize=1.8,
                capthick=0.6,
                zorder=3,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    style_axes(ax)
    if compact:
        ax.set_ylabel("Significant (%)", labelpad=1.5)
        ax.set_ylim(0, 43)
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", pad=1)
    add_statistics(
        ax,
        positions,
        intervals,
        chance_p,
        comparison_p,
        comparison_label="symbol" if compact else "pvalue",
        compact=compact,
    )
    if legend_side:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.52),
            ncol=1,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.35,
            borderaxespad=0,
            fontsize=LEGEND_LABEL_PT,
        )
        fig.subplots_adjust(left=0.13, right=0.84, bottom=0.28, top=0.88)
    else:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=2,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.75,
            borderaxespad=0,
            fontsize=LEGEND_LABEL_PT,
        )
        fig.subplots_adjust(left=0.29, right=0.98, bottom=0.20, top=0.80)

    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    return [png_path, svg_path]


def save_statistics(path: Path, statistics: dict[str, object]) -> None:
    proportions = np.asarray(statistics["proportions"])
    intervals = np.asarray(statistics["intervals"])
    chance_p = np.asarray(statistics["chance_p"])
    comparison_p = np.asarray(statistics["comparison_p"])
    payload = {
        "figure_size_in": list(FIGSIZE),
        "chance_rate": CHANCE,
        "chance_test": "one-sided exact binomial test, alternative > 0.05",
        "confidence_interval": "two-sided 95% Clopper-Pearson exact binomial",
        "before_after_test": "two-sided Fisher exact test on unmatched aggregate counts",
        "conditions": [],
        "before_after_comparisons": [],
        "source_discrepancy": (
            "The note reports 19.80% for Probe 2 before retraction, but "
            "20/102 equals 19.61%; count-derived values are plotted."
        ),
        "statistics_note": str((Path(__file__).parent / "STATISTICS_NOTE.md").resolve()),
    }
    for probe in range(2):
        for phase in range(2):
            payload["conditions"].append({
                "probe": PROBES[probe],
                "phase": PHASES[phase],
                "significant": int(COUNTS[probe, phase]),
                "total": int(TOTALS[probe, phase]),
                "percent": float(100 * proportions[probe, phase]),
                "ci95_percent": (100 * intervals[probe, phase]).tolist(),
                "exact_binomial_p_vs_5_percent": float(chance_p[probe, phase]),
            })
        payload["before_after_comparisons"].append({
            "probe": PROBES[probe],
            "fisher_exact_two_sided_p": float(comparison_p[probe]),
        })
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


