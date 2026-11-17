from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

import re

SOURCE_DIR = HERE / "DTA"

PULSE_COUNTS_M = [0, 1, 2, 10, 35, 45, 65]
COLORS = ["#202020", "#3b6f8f", "#4f8a72", "#9a7b37", "#9a5552", "#735c8e", "#a25478"]

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def natural_key(path: Path) -> int:
    match = re.match(r"(\d+)", path.name)
    return int(match.group(1)) if match else 999


def parse_gamry_eis(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    zcurve_index = next(i for i, line in enumerate(lines) if line.startswith("ZCURVE"))
    rows: list[dict[str, float]] = []
    for line in lines[zcurve_index + 3 :]:
        if not line.startswith("\t") or not line.strip():
            break
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append(
            {
                "point": int(float(parts[0])),
                "time_s": float(parts[1]),
                "frequency_hz": float(parts[2]),
                "zreal_ohm": float(parts[3]),
                "zimag_ohm": float(parts[4]),
                "zmod_ohm": float(parts[6]),
                "phase_deg": float(parts[7]),
            }
        )
    if not rows:
        raise ValueError(f"No ZCURVE observations in {path}")
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE_DIR.glob("*EIS*.DTA"), key=natural_key)
    if len(files) != len(PULSE_COUNTS_M):
        raise RuntimeError(f"Expected 7 EIS files, found {len(files)} in {SOURCE_DIR}")

    spectra: list[pd.DataFrame] = []
    one_khz_rows: list[dict[str, object]] = []
    for pulse_count_m, path in zip(PULSE_COUNTS_M, files):
        frame = parse_gamry_eis(path)
        frame.insert(0, "pulse_count_millions", pulse_count_m)
        frame.insert(1, "pulse_label", f"{pulse_count_m}M")
        frame.insert(2, "electrode_id", "EBL82-2-8")
        frame.insert(3, "source_file", path.name)
        frame.insert(4, "source_path", path.name)
        spectra.append(frame)
        nearest = frame.iloc[(frame["frequency_hz"] - 1000.0).abs().argmin()]
        one_khz_rows.append(
            {
                "pulse_count_millions": pulse_count_m,
                "pulse_label": f"{pulse_count_m}M",
                "electrode_id": "EBL82-2-8",
                "frequency_hz": nearest["frequency_hz"],
                "zmod_kohm": nearest["zmod_ohm"] / 1000.0,
                "phase_deg": nearest["phase_deg"],
                "source_file": path.name,
                "source_path": path.name,
            }
        )

    all_spectra = pd.concat(spectra, ignore_index=True)
    one_khz = pd.DataFrame(one_khz_rows)
    all_spectra.to_csv(OUT_DIR / "SF04a_EIS_same_electrode_all_frequency_points.csv", index=False)
    one_khz.to_csv(OUT_DIR / "SF04a_EIS_same_electrode_1kHz_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for color, (pulse_count_m, frame) in zip(COLORS, all_spectra.groupby("pulse_count_millions", sort=False)):
        ax.plot(
            frame["frequency_hz"],
            frame["zmod_ohm"] / 1000.0,
            color=color,
            linewidth=0.85,
            label=f"{pulse_count_m:g}M",
        )
        nearest = frame.iloc[(frame["frequency_hz"] - 1000.0).abs().argmin()]
        ax.scatter(
            nearest["frequency_hz"],
            nearest["zmod_ohm"] / 1000.0,
            s=6,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    ax.axvline(1000.0, color="#777777", linestyle=":", linewidth=0.45, alpha=0.45, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance magnitude (kOhm)")
    ax.legend(title="Cumulative pulses", title_fontsize=6, frameon=False, ncol=2, handlelength=1.5)
    style_axis(ax)
    save_figure(fig, OUT_DIR / "SF04a_EIS_same_electrode_0_to_65M_magnitude")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    OUT_DIR = parser.parse_args().out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
