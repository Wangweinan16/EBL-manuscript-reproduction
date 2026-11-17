"""Regenerate threshold and modulation panels from session-level observations."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent

import matplotlib as mpl
import pandas as pd

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

PANEL_CONFIGS = {
    "b": {
        "event_metric": "current",
        "mean_col": "mean_activation_current_uA",
        "sd_col": "sd_activation_current_uA",
        "ylabel": "Activation current (uA)",
        "hist_mean_xlabel": "Mean activation current (uA)",
        "hist_sd_xlabel": "SD activation current (uA)",
        "color": "#a9d8ee",
        "edge": "#246b8f",
    },
    "c": {
        "event_metric": "signed_modulation_index_pre2s",
        "mean_col": "mean_signed_modulation_index",
        "sd_col": "sd_signed_modulation_index",
        "ylabel": "Modulation index",
        "hist_mean_xlabel": "Mean modulation index",
        "hist_sd_xlabel": "SD modulation index",
        "color": "#f0a188",
        "edge": "#a6402b",
    },
}

def savefig(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 500})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def format_pair_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.4, 51.6)
    ax.set_xticks(np.arange(1, 52, 5))
    ax.set_xlabel("Unit-contact pair")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.5, pad=1)


def plot_pair_mean_sd(panel: str, merged: pd.DataFrame, pair_stats: pd.DataFrame) -> None:
    cfg = PANEL_CONFIGS[panel]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x = pair_stats["pair_index"].to_numpy(dtype=float)
    mean = pair_stats[cfg["mean_col"]].to_numpy(dtype=float)
    sd = pair_stats[cfg["sd_col"]].to_numpy(dtype=float)
    ax.bar(x, mean, width=0.72, color=cfg["color"], edgecolor=cfg["edge"], linewidth=0.3, zorder=2)
    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="black", elinewidth=0.45, capsize=1.1, zorder=3)
    metric = cfg["event_metric"]
    for pair_index, group in merged.groupby("pair_index", sort=False):
        offsets = np.linspace(-0.24, 0.24, len(group)) if len(group) > 1 else np.array([0.0])
        ax.scatter(pair_index + offsets, group[metric], s=3, color="black", alpha=0.65, linewidths=0, zorder=4)
    ax.set_ylabel(cfg["ylabel"])
    format_pair_axis(ax)
    if panel == "c":
        ax.axhline(0, color="0.45", lw=0.5, zorder=1)
        ax.set_ylim(-0.05, 1.05)
    savefig(fig, UPDATED / f"SF10{panel}_51pairs_mean_sd_points_updated_metric")


def plot_histograms(panel: str, pair_stats: pd.DataFrame) -> None:
    cfg = PANEL_CONFIGS[panel]
    specs = [
        (cfg["mean_col"], cfg["hist_mean_xlabel"], f"SF10{panel}_hist_distribution_pair_means_updated_metric"),
        (cfg["sd_col"], cfg["hist_sd_xlabel"], f"SF10{panel}_hist_distribution_pair_sds_updated_metric"),
    ]
    for col, xlabel, stem in specs:
        fig, ax = plt.subplots(figsize=(2.2, 1.65))
        ax.hist(pair_stats[col].dropna(), bins=12, color="black", linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.2, width=0.5, pad=1)
        savefig(fig, UPDATED / stem)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    UPDATED = parser.parse_args().out
    UPDATED.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(HERE / 'observations.csv').sort_values(['pair_index', 'trial'])
    assert len(events) == 275 and not events.duplicated(['pair_index','trial']).any()
    stats = pd.read_csv(HERE / 'pair_statistics.csv').sort_values('pair_index')
    assert len(stats) == 51
    rates = events['stim_train_rate_Hz']
    baseline = events['baseline_pre2s_rate_Hz']
    np.testing.assert_allclose(events['signed_modulation_index_pre2s'], (rates-baseline)/(rates+baseline))
    for panel, cfg in PANEL_CONFIGS.items():
        agg = events.groupby('pair_index')[cfg['event_metric']].agg(['mean','std'])
        np.testing.assert_array_equal(agg.index, stats['pair_index'])
        np.testing.assert_allclose(agg['mean'], stats[cfg['mean_col']])
        np.testing.assert_allclose(agg['std'], stats[cfg['sd_col']], atol=1e-12)
        plot_pair_mean_sd(panel, events, stats)
        plot_histograms(panel, stats)
