from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SORT_ROOT = SCRIPT_DIR / "fig2e_sort_20min_si099_primary"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_figure2_panels"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache_figure2_panels"
ANIMALS = ("EBL2", "EBL6", "EBL7", "EBL8", "EBL9")
LAYOUTS = (43, 64, 85, 106, 128)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, png: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", format="svg", bbox_inches="tight", transparent=True)
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight", transparent=True)
    if png:
        fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"saved {output_dir / f'{stem}.svg'}")
    print(f"saved {output_dir / f'{stem}.pdf'}")


def matrix_from_summary(summary_rows: list[dict], value_key: str) -> np.ndarray:
    mat = np.full((len(LAYOUTS), len(ANIMALS)), np.nan, dtype=float)
    animal_idx = {animal: idx for idx, animal in enumerate(ANIMALS)}
    layout_idx = {layout: idx for idx, layout in enumerate(LAYOUTS)}
    for row in summary_rows:
        mat[layout_idx[int(row["layout"])], animal_idx[str(row["animal"])]] = float(row[value_key])
    return mat


def localized_matrix_from_units(unit_rows: list[dict]) -> np.ndarray:
    counts_by_layout = []
    for layout in LAYOUTS:
        vals = [
            float(row["localized_electrodes_20pct"])
            for row in unit_rows
            if int(row["layout"]) == layout and row["status"] == "accepted"
        ]
        counts_by_layout.append(np.asarray(vals, dtype=float))
    max_n = max((v.size for v in counts_by_layout), default=0)
    mat = np.full((len(LAYOUTS), max_n), np.nan, dtype=float)
    for i, vals in enumerate(counts_by_layout):
        mat[i, : vals.size] = vals
    return mat


def style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=5, direction="out", length=3, width=0.8, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def equal_spaced_offsets_by_value(vals: np.ndarray, width: float = 5.2) -> np.ndarray:
    """Spread identical y-values evenly so integer-count points are visible."""
    offsets = np.zeros(vals.size, dtype=float)
    for value in np.unique(vals):
        idx = np.flatnonzero(vals == value)
        if idx.size > 1:
            offsets[idx] = np.linspace(-width / 2, width / 2, idx.size)
    return offsets


def plot_yield(summary_rows: list[dict], output_dir: Path, png: bool) -> None:
    x = np.asarray(LAYOUTS, dtype=float)
    values = matrix_from_summary(summary_rows, "accepted_count")
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)

    fig, ax = plt.subplots(figsize=(1.82, 1.86))
    ax.fill_between(x, mean - std, mean + std, color="0.68", alpha=0.9, linewidth=0)
    ax.plot(x, mean, color="0.25", lw=1.0)
    for idx, layout in enumerate(x):
        vals = values[idx]
        vals = vals[np.isfinite(vals)]
        jitter = np.linspace(-1.8, 1.8, vals.size) if vals.size > 1 else np.zeros(vals.size)
        ax.scatter(layout + jitter, vals, color="black", s=2.0, alpha=1.0, linewidths=0, zorder=2)
    ax.set_xlabel("Electrode layouts", fontsize=7, labelpad=1)
    ax.set_ylabel("Single unit yield", fontsize=7, labelpad=2)
    ax.set_xlim(38, 133)
    ax.set_xticks([40, 60, 80, 100, 120])
    ymax = max(110, float(np.nanmax(values)) + 8)
    ax.set_ylim(15, ymax)
    style_axis(ax)
    save_figure(fig, output_dir, "fig2e_single_unit_yield", png=png)


def plot_localized(unit_rows: list[dict], output_dir: Path, png: bool, stem: str) -> None:
    x = np.asarray(LAYOUTS, dtype=float)
    values = localized_matrix_from_units(unit_rows)
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)

    fig, ax = plt.subplots(figsize=(1.82, 1.86))
    ax.fill_between(x, mean - std, mean + std, color="0.68", alpha=0.9, linewidth=0)
    ax.plot(x, mean, color="0.25", lw=1.0)
    for idx, layout in enumerate(x):
        vals = values[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            offsets = equal_spaced_offsets_by_value(vals, width=5.2)
            ax.scatter(layout + offsets, vals, color="0.35", alpha=0.65, s=2.0, linewidths=0, zorder=2)
    ax.set_xlabel("Electrode layouts", fontsize=7, labelpad=1)
    ax.set_ylabel("Localized contacts", fontsize=7, labelpad=2)
    ax.set_xlim(38, 133)
    ax.set_xticks([40, 60, 80, 100, 120])
    ymax = max(15, float(np.nanmax(values)) + 1 if np.isfinite(values).any() else 15)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, np.ceil(ymax) + 1, 2))
    style_axis(ax)
    save_figure(fig, output_dir, stem, png=png)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Fig 2B from curated animal/layout counts and units.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    with (SCRIPT_DIR / "units.csv").open(newline="") as f:
        units = list(csv.DictReader(f))
    with (SCRIPT_DIR / "sessions.csv").open(newline="") as f:
        sessions = list(csv.DictReader(f))
    plot_yield(sessions, args.output_dir, png=True)
    plot_localized(units, args.output_dir, png=True, stem="Fig2B_localized_contacts")
