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

from scipy.stats import ttest_rel
CURRENT_VALUES = np.arange(1, 6)
COLORS = {
    "EBL2": "#277DA1",
    "EBL6": "#43AA8B",
    "EBL7": "#F9C74F",
    "EBL8": "#F3722C",
    "EBL9": "#8E5EA2",
}
ANIMAL_ORDER = ["EBL9", "EBL2", "EBL6", "EBL7", "EBL8"]

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)


def savefig(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 500})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_complete_grid(channel_counts: pd.DataFrame, animal_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    animals = [animal for animal in ANIMAL_ORDER if animal in set(channel_counts["animal"])]
    offsets = np.linspace(-0.24, 0.24, len(animals))
    rng = np.random.default_rng(905)
    for animal, offset in zip(animals, offsets):
        color = COLORS[animal]
        animal_points = channel_counts[channel_counts["animal"] == animal]
        for current, points in animal_points.groupby("current_uA"):
            x = current + offset + rng.uniform(-0.026, 0.026, len(points))
            ax.scatter(
                x,
                points["activated_unit_count"],
                s=4.2,
                facecolors=color,
                edgecolors="none",
                alpha=0.55,
                zorder=1,
            )
        summary = animal_summary[
            (animal_summary["animal"] == animal)
            & (animal_summary["aggregation"] == "all_channel_session_grid")
        ].sort_values("current_uA")
        ax.errorbar(
            summary["current_uA"] + offset, summary["mean"], yerr=summary["sem"],
            marker="o", markersize=2.5, linewidth=0.9, elinewidth=0.55,
            capsize=1.5, color=color, label=animal, zorder=3,
        )
    ax.set_xlabel("Stimulation current (μA)")
    ax.set_ylabel("Activated units per stimulation channel")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.62, 5.38)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=5.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    savefig(fig, PUB / "SF07b_grouped_ebl9_left")

def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return "ns"


def render_panel(
    summary: pd.DataFrame,
    out_base: Path,
    title: str | None = None,
    errorbar: str = "sd",
    figsize: tuple[float, float] = (1.58, 1.9),
    axis_label_fontsize: float = 7.0,
    tick_fontsize: float = 6.0,
    legend_fontsize: float = 5.8,
    sig_fontsize: float = 5.5,
    show_legend: bool = True,
) -> None:
    summary = summary.sort_values("current_uA")
    currents = summary["current_uA"].to_numpy(dtype=float)
    stim_mean = summary["stim_rate_mean_hz"].to_numpy(dtype=float)
    baseline_mean = summary["baseline_rate_mean_hz"].to_numpy(dtype=float)
    if errorbar == "sem":
        stim_err = summary["stim_rate_sem_hz"].to_numpy(dtype=float)
        baseline_err = summary["baseline_rate_sem_hz"].to_numpy(dtype=float)
    elif errorbar == "sd":
        stim_err = summary["stim_rate_sd_hz"].to_numpy(dtype=float)
        baseline_err = summary["baseline_rate_sd_hz"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown errorbar: {errorbar}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        currents,
        stim_mean,
        yerr=stim_err,
        color="#2F49C8",
        ecolor="#2F49C8",
        linewidth=0.75,
        elinewidth=0.45,
        capsize=1.4,
        marker=None,
        label="Modulated firing rate",
    )
    ax.errorbar(
        currents,
        baseline_mean,
        yerr=baseline_err,
        color="#E41A1C",
        ecolor="#E41A1C",
        linewidth=0.75,
        elinewidth=0.45,
        capsize=1.4,
        marker=None,
        label="Baseline firing rate",
    )
    y_top = max(float(np.nanmax(stim_mean + stim_err)), float(np.nanmax(baseline_mean + baseline_err)), 1.0)
    y_lim_top = y_top * 1.18
    for _, row in summary.iterrows():
        label = significance_label(float(row["paired_ttest_p"]))
        current = float(row["current_uA"])
        stim_key = "stim_rate_sem_hz" if errorbar == "sem" else "stim_rate_sd_hz"
        baseline_key = "baseline_rate_sem_hz" if errorbar == "sem" else "baseline_rate_sd_hz"
        y = max(float(row["stim_rate_mean_hz"] + row[stim_key]), float(row["baseline_rate_mean_hz"] + row[baseline_key]))
        label_y = min(y + 0.045 * y_top, y_lim_top - 0.085 * y_top)
        if current <= 1.1:
            label_x = 1.08
        elif current >= 4.9:
            label_x = 4.92
        else:
            label_x = current
        ax.text(label_x, label_y, label, fontsize=sig_fontsize, ha="center", va="bottom", clip_on=True)

    ax.set_xlabel("Current (μA)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Firing rate (Hz)", fontsize=axis_label_fontsize)
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(0, y_lim_top)
    ax.set_xticks(CURRENT_VALUES)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=0.5, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=6, pad=2)
    if show_legend:
        ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper left", handlelength=1.1, handletextpad=0.4)
    fig.savefig(out_base.with_suffix(".svg"), format="svg", bbox_inches="tight", transparent=True)
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight", transparent=False)
    plt.close(fig)

def main():
    for panel, unit, channel in [('A', 142, 94), ('B', 145, 117)]:
        trials = pd.read_csv(HERE / f'unit{unit}_trial_rates.csv')
        summary = pd.read_csv(HERE / f'unit{unit}_summary.csv').sort_values('current_uA')
        for row in summary.itertuples():
            group = trials[trials.current_uA == row.current_uA]
            assert len(group) == row.n_trials == 10
            np.testing.assert_allclose(ttest_rel(group.stim_rate_hz, group.baseline_rate_hz).pvalue, row.paired_ttest_p)
            np.testing.assert_allclose(group.stim_rate_hz.sem(), row.stim_rate_sem_hz)
        render_panel(summary, PUB / ('S08'+panel), title=f'Unit {unit} / Ch {channel}', errorbar='sem', figsize=(1.68,1.75), show_legend=(panel == "A"))
    grid = pd.read_csv(HERE / 'per_stim_channel_counts_complete_grid.csv')
    events = pd.read_csv(HERE / 'offset_corrected_activation_events.csv')
    keys = ['animal','session_date','current_uA','stim_channel']
    observed = events.groupby(keys).unit_id.nunique().reindex(pd.MultiIndex.from_frame(grid[keys]),fill_value=0)
    np.testing.assert_array_equal(observed,grid.activated_unit_count)
    summary = grid.groupby(['animal','current_uA']).activated_unit_count.agg(['count','mean','std','sem']).reset_index()
    assert len(grid) == 600 and (summary['count'] == 24).all()
    summary['aggregation'] = 'all_channel_session_grid'
    plot_complete_grid(grid, summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    PUB = parser.parse_args().out
    PUB.mkdir(parents=True, exist_ok=True)
    main()
