from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "20240508" / "20240508"
USEFUL_CV_DIR = DATA_DIR / "useful CV"
USEFUL_EIS_DIR = DATA_DIR / "useful EIS"
OUT_DIR = ROOT / "cv_csc_msc_cic_results"
PLOTS_DIR = Path("outputs")

AREA_CM2 = 2.25e-6
CATHODIC_LIMIT_V = -0.6
ANODIC_LIMIT_V = 0.8


@dataclass(frozen=True)
class CvSpec:
    filename: str
    label: str
    pulses_millions: float


@dataclass(frozen=True)
class MscSpec:
    filename: str
    label: str
    nominal_current_ua: float


CV_SEQUENCE = [
    CvSpec("1CV-EBL82-2-8-0.csv", "0 pulses", 0),
    CvSpec("2CV-EBL82-2-8-Post 1M-1.csv", "1M pulses", 1),
    CvSpec("3CV-EBL82-2-8-Post 1M-2.csv", "2M pulses", 2),
    CvSpec("4CV-EBL82-2-8-Post 7M-10.csv", "10M pulses", 10),
    CvSpec("5CV-EBL82-2-8-Post 25M-35.csv", "35M pulses", 35),
    CvSpec("6CV-EBL82-2-8-Post 10M-45.csv", "45M pulses", 45),
    CvSpec("7CV-EBL82-2-8-Post 17M-65.csv", "65M pulses", 65),
]

MSC_SEQUENCE = [
    MscSpec("MSC-EBL82-2-8-0-3uA.csv", "3 uA", 3),
    MscSpec("MSC-EBL82-2-8-0-6uA.csv", "6 uA", 6),
    MscSpec("MSC-EBL82-2-8-0-9uA.csv", "9 uA", 9),
    MscSpec("MSC-EBL82-2-8-0-12uA.csv", "12 uA", 12),
    MscSpec("MSC-EBL82-2-8-0-15uA.csv", "15 uA", 15),
    MscSpec("MSC-EBL82-2-8-0-18uA.csv", "18 uA", 18),
]


def read_gamry_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a Gamry CSV export with a second units row."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("Pt") != "#"]
    out: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                continue
            try:
                out.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def save_current_figure(name: str) -> None:
    for suffix in (".png", ".svg"):
        plt.savefig(PLOTS_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_sequence() -> None:
    plt.figure(figsize=(7.0, 5.0))
    for spec in CV_SEQUENCE:
        data = read_gamry_csv(USEFUL_CV_DIR / spec.filename)
        plt.plot(data["Vf"], moving_average(data["Im"]) * 1e9, linewidth=2, label=spec.label)
    plt.xlim(-1.0, 1.2)
    plt.xlabel("Potential vs. Ag|AgCl (V)")
    plt.ylabel("Current (nA)")
    plt.title("CV sequence after repeated pulsing")
    plt.legend(frameon=False, fontsize=9, loc="lower right")
    plt.box(True)
    save_current_figure("cv_sequence_after_pulsing")


def plot_msc() -> None:
    traces = []
    ref_transition_us = None
    ref_initial_v = None
    for spec in MSC_SEQUENCE:
        data = read_gamry_csv(DATA_DIR / spec.filename)
        t_us = data["T"] * 1e6
        v = data["Vf"].copy()
        if ref_initial_v is None:
            ref_initial_v = float(v[0])
        else:
            v = v - (float(v[0]) - ref_initial_v)
        transition = np.flatnonzero(v < (-0.01 if spec.nominal_current_ua > 3 else 0.01))
        transition_us = float(t_us[transition[0]]) if transition.size else float(t_us[0])
        if ref_transition_us is None:
            ref_transition_us = transition_us
        traces.append((spec, t_us - (transition_us - ref_transition_us), v, data["Im"] * 1e6))

    plt.figure(figsize=(7.0, 5.0))
    for spec, t_us, v, _ in traces:
        plt.plot(t_us, v, linewidth=2, label=spec.label)
    plt.xlim(0, 400)
    plt.ylim(-2, 1.5)
    plt.xlabel("Time (us)")
    plt.ylabel("Potential vs. Ag|AgCl (V)")
    plt.title("MSC voltage transients")
    plt.legend(frameon=False, fontsize=9, loc="lower right")
    plt.box(True)
    save_current_figure("msc_voltage_transients")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render representative voltage transients and cycling CV curves.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    PLOTS_DIR = parser.parse_args().output_dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_cv_sequence()
    plot_msc()
