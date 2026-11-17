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

IMPLANT_DATE = "2023-09-28"
EXCLUDED_SESSION_INDEX = 1
COLORS = {"p2p": "#2166ac", "impedance": "#b45309", "points": "#6b7280"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def deterministic_jitter(n: int, width_days: float) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width_days, width_days, n)


def style_axis(
    ax: plt.Axes,
    xlim: tuple[float, float] = (10, 185),
    ticks: list[int] = [30, 60, 90, 120, 150, 180],
    label: str = "Time after implantation (days)",
) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel(label)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    configure_style()
    

    units = pd.read_csv(HERE / "SF06_unit_level_waveform_metrics.csv")
    impedance = pd.read_csv(HERE / "SF06_channel_impedance_long.csv")
    sessions = pd.read_csv(HERE / "SF06_session_summary.csv")

    sessions = sessions[sessions["session_index"] != EXCLUDED_SESSION_INDEX].copy()
    sessions["day_after_implant"] = sessions["day_post_implant_provisional"].astype(int)
    day_map = sessions.set_index("session_index")["day_after_implant"].to_dict()
    units = units[units["session_index"] != EXCLUDED_SESSION_INDEX].copy()
    units["day_after_implant"] = units["session_index"].map(day_map)
    impedance = impedance[impedance["session_index"] != EXCLUDED_SESSION_INDEX].copy()
    impedance["day_after_implant"] = impedance["session_index"].map(day_map)
    days = sessions["day_after_implant"].to_numpy(float)

    units.to_csv(FIGURES / "SF06_unit_level_waveform_metrics_days.csv", index=False)
    impedance.to_csv(FIGURES / "SF06_channel_impedance_days.csv", index=False)
    sessions.to_csv(FIGURES / "SF06_session_summary_days.csv", index=False)

    fig, ax = plt.subplots(figsize=(1.75, 1.45))
    medians, q1s, q3s = [], [], []
    for day in days:
        values = units.loc[
            units["day_after_implant"] == day, "spike_p2p_median_uv"
        ].dropna().to_numpy(float)
        ax.scatter(
            day + deterministic_jitter(len(values), 1.4),
            values,
            s=3.2,
            color=COLORS["points"],
            alpha=0.22,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )
        medians.append(np.median(values))
        q1s.append(np.quantile(values, 0.25))
        q3s.append(np.quantile(values, 0.75))
    ax.fill_between(days, q1s, q3s, color=COLORS["p2p"], alpha=0.14, linewidth=0)
    ax.plot(days, medians, color=COLORS["p2p"], marker="o", markersize=2.6, linewidth=1.0)
    ax.set_ylabel("Median spike P2P (μV)")
    style_axis(ax)
    save(fig, "SF06a_median_spike_p2p_days_exclude_week2")

    fig, ax = plt.subplots(figsize=(1.75, 1.45))
    for day in days:
        values = impedance.loc[
            impedance["day_after_implant"] == day, "impedance_MOhm"
        ].dropna().to_numpy(float)
        ax.scatter(
            day + deterministic_jitter(len(values), 1.4),
            values,
            s=2.8,
            color=COLORS["points"],
            alpha=0.20,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )
    ax.fill_between(
        days,
        sessions["impedance_q1_MOhm"],
        sessions["impedance_q3_MOhm"],
        color=COLORS["impedance"],
        alpha=0.14,
        linewidth=0,
    )
    ax.plot(
        days,
        sessions["impedance_median_MOhm"],
        color=COLORS["impedance"],
        marker="o",
        markersize=2.6,
        linewidth=1.0,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.01, 30)
    ax.set_ylabel("Impedance at 1 kHz (MΩ)")
    style_axis(ax)
    save(fig, "SF06b_impedance_days_exclude_week2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    FIGURES = parser.parse_args().out
    FIGURES.mkdir(parents=True, exist_ok=True)
    main()
