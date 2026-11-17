# -*- coding: utf-8 -*-
"""Reproduce Supplementary Figure 6 / Figure 2g tracking-location stability.

The original supplementary panel is the ``std_x`` versus ``std_y`` summary for
unit-location estimates across the 10 consecutive tracking days. It is now also
used as manuscript Figure 2g, with the original Figure 2g connectivity map
moving to Figure 2h. This script uses the same cached SpikeInterface waveform
folders as the maintained Figure 3 / current Figure 2f tracking-map runner, but
avoids the old notebook state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WAVEFORM_ROOT = Path(__file__).resolve().parent / 'data'
DEFAULT_OUTPUT_DIR = Path('outputs')
PANEL_SIZE_IN = (1.02, 1.02)
AXIS_LABEL_SIZE_PT = 7
TICK_LABEL_SIZE_PT = 6


def load_day(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    metrics = pd.read_csv(folder / "quality_metrics" / "metrics.csv", index_col=0)
    unit_ids = metrics.index.astype(int).to_numpy()
    locations = np.load(folder / "unit_locations" / "unit_locations.npy")
    if len(unit_ids) != len(locations):
        raise ValueError(f"Unit/location count mismatch in {folder}")
    return unit_ids, locations


def collect_location_stability(waveform_root: Path) -> pd.DataFrame:
    day_data = [load_day(waveform_root / f"waveforms_sorting_{day}") for day in range(1, 11)]
    common_units = sorted(set(day_data[0][0]).intersection(*(set(ids) for ids, _ in day_data[1:])))
    rows = []
    for unit_id in common_units:
        locations = []
        for ids, locs in day_data:
            index = int(np.flatnonzero(ids == unit_id)[0])
            locations.append(locs[index, :2])
        arr = np.asarray(locations, dtype=float)
        rows.append(
            {
                "unit_id": unit_id,
                "n_days": arr.shape[0],
                "mean_x_um": float(np.mean(arr[:, 0])),
                "mean_y_um": float(np.mean(arr[:, 1])),
                "std_x_um": float(np.std(arr[:, 0], ddof=0)),
                "std_y_um": float(np.std(arr[:, 1], ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def draw_panel(table: pd.DataFrame, output_prefix: Path) -> None:
    x = table["std_x_um"].to_numpy(dtype=float)
    y = table["std_y_um"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN)
    magenta = "#ec008c"
    ax.scatter(x, y, s=2.4, color=magenta, alpha=0.6, edgecolors="none")

    try:
        from scipy.stats import gaussian_kde

        values = np.vstack([x, y])
        kde = gaussian_kde(values)
        xi, yi = np.mgrid[0:10:200j, 0:10:200j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        point_density = kde(values)
        level = np.percentile(point_density, 10)
        ax.contour(xi, yi, zi, levels=[level], colors=[magenta], linewidths=0.45)
    except Exception as exc:
        print(f"Skipping KDE contour because scipy/KDE failed: {exc}")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("std_x (um)", fontsize=AXIS_LABEL_SIZE_PT, labelpad=1.0)
    ax.set_ylabel("std_y (um)", fontsize=AXIS_LABEL_SIZE_PT, labelpad=1.0)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE_PT, width=0.5, length=1.6, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    fig.subplots_adjust(left=0.32, bottom=0.30, right=0.96, top=0.96)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, transparent=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), transparent=True)
    plt.close(fig)


def write_manifest(output_dir: Path, waveform_root: Path, table: pd.DataFrame) -> None:
    manifest = {
        "waveform_root": str(waveform_root),
        "n_units_common_to_all_10_days": int(len(table)),
        "median_std_x_um": float(table["std_x_um"].median()),
        "median_std_y_um": float(table["std_y_um"].median()),
        "fraction_std_x_lt_6_and_std_y_lt_8": float(((table["std_x_um"] < 6) & (table["std_y_um"] < 8)).mean()),
        "manuscript_panel": "Figure 2g",
        "superseded_panel": "Supplementary Figure 6",
        "panel_size_in": list(PANEL_SIZE_IN),
        "axis_label_size_pt": AXIS_LABEL_SIZE_PT,
        "tick_label_size_pt": TICK_LABEL_SIZE_PT,
        "svg_text_policy": "matplotlib svg.fonttype=none; text remains editable SVG text",
        "source_files_per_day": [
            str(waveform_root / f"waveforms_sorting_{day}" / "unit_locations" / "unit_locations.npy")
            for day in range(1, 11)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-root", type=Path, default=DEFAULT_WAVEFORM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = collect_location_stability(args.waveform_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "s6_tracking_location_stability.csv", index=False)
    draw_panel(table, args.output_dir / "s6_tracking_location_stability")
    draw_panel(table, args.output_dir / "fig2g_s6_tracking_location_stability_1p02in")
    write_manifest(args.output_dir, args.waveform_root, table)
    print(f"Saved S6 tracking stability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
