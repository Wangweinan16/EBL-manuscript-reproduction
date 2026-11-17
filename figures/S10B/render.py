from __future__ import annotations
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

UNIT_ID = 253
STIM_CHANNEL = 121
SESSION_INDICES = list(range(1, 8))
PULSES_PER_TRIAL = 50
PRE_MS = 2.0
POST_MS = 8.0
BIN_MS = 0.25
SIGMA_MS = 0.5
KERNEL_TRUNCATION_SIGMA = 3.0
EXTRACTION_PRE_MS = PRE_MS + KERNEL_TRUNCATION_SIGMA * SIGMA_MS
EXTRACTION_POST_MS = POST_MS + KERNEL_TRUNCATION_SIGMA * SIGMA_MS
ARTIFACT_START_MS = -0.4
ARTIFACT_END_MS = 2.4
PULSE_INTERVAL_MS = 10.0
NEXT_ARTIFACT_START_MS = PULSE_INTERVAL_MS + ARTIFACT_START_MS
FULL_SUPPORT_END_MS = NEXT_ARTIFACT_START_MS - KERNEL_TRUNCATION_SIGMA * SIGMA_MS

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
        target = stem.with_suffix(suffix)
        if os.name == "nt" and len(str(target.resolve())) >= 248:
            temporary = target.parent / f".__sf10a_export{suffix}"
            fig.savefig(temporary, bbox_inches="tight", **kwargs)
            os.replace(temporary, "\\\\?\\" + str(target.resolve()))
        else:
            fig.savefig(target, bbox_inches="tight", **kwargs)
    plt.close(fig)


def edge_corrected_rate(
    relative_times_ms: np.ndarray, pulse_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-bin counts and artifact-separated, edge-corrected rate."""
    bins = np.arange(-EXTRACTION_PRE_MS, EXTRACTION_POST_MS + BIN_MS, BIN_MS)
    histogram, edges = np.histogram(relative_times_ms, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_extent = KERNEL_TRUNCATION_SIGMA * SIGMA_MS
    kernel_range = np.arange(-kernel_extent, kernel_extent + BIN_MS, BIN_MS)
    kernel = np.exp(-(kernel_range**2) / (2 * SIGMA_MS**2))
    kernel /= kernel.sum()

    smoothed_counts = np.full(histogram.shape, np.nan, dtype=float)
    pre_mask = centers < ARTIFACT_START_MS
    post_mask = centers > ARTIFACT_END_MS
    for segment_mask in (pre_mask, post_mask):
        numerator = np.convolve(histogram * segment_mask, kernel, mode="same")
        denominator = np.convolve(segment_mask.astype(float), kernel, mode="same")
        usable = segment_mask & (denominator > 0)
        smoothed_counts[usable] = numerator[usable] / denominator[usable]

    rate_hz = smoothed_counts / (pulse_count * BIN_MS / 1000.0)
    return centers, histogram.astype(float), rate_hz


def main():
    colors = plt.get_cmap("tab20").colors[:len(SESSION_INDICES)]
    all_raster = pd.read_csv(HERE / "raster.csv")
    all_rates = pd.read_csv(HERE / "rates.csv")
    scan = pd.read_csv(HERE / "current_scan.csv")
    pulses = pd.read_csv(HERE / "pulses.csv")
    selected = scan.loc[scan.max_postartifact_edge_corrected_rate_Hz.idxmax(), 'current_uA']
    assert selected == 4
    for idx in SESSION_INDICES:
        rows = all_raster[all_raster.session_index == idx]
        reference = all_rates[all_rates.session_index == idx]
        n = len(pulses[(pulses.session_index == idx) & (pulses.current_uA == selected)])
        assert n == 500
        # Compute milliseconds directly from 30 kHz acquisition samples to preserve exact bin-edge behavior.
        times = (rows.spike_sample-rows.pulse_sample).to_numpy() * 1000.0 / 30000.0
        centers, counts, rate = edge_corrected_rate(times,n)
        np.testing.assert_array_equal(counts,reference['raw_pooled_count_per_0.25ms_bin'])
        np.testing.assert_allclose(rate, reference.edge_corrected_rate_Hz, equal_nan=True)
    fig, ax_raster = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)
    ax_rate = ax_raster.twinx()
    for session_index, color in zip(SESSION_INDICES, colors):
        raster = all_raster[all_raster["session_index"] == session_index]
        rate = all_rates[all_rates["session_index"] == session_index]
        raster_visible = (
            (raster["relative_time_ms"] >= ARTIFACT_START_MS)
            & (raster["relative_time_ms"] < NEXT_ARTIFACT_START_MS)
        )
        ax_raster.scatter(
            raster.loc[raster_visible, "relative_time_ms"],
            raster.loc[raster_visible, "pulse_index"],
            marker="|",
            s=2.2,
            linewidths=0.35,
            color=color,
            alpha=0.8,
            zorder=2,
        )
        smooth_visible = (
            (rate["time_ms"] > ARTIFACT_END_MS)
            & (rate["time_ms"] < NEXT_ARTIFACT_START_MS)
        )
        ax_rate.plot(
            rate.loc[smooth_visible, "time_ms"],
            rate.loc[smooth_visible, "edge_corrected_rate_Hz"],
            color=color,
            linewidth=0.8,
            label=f"Session {session_index}",
            zorder=4,
        )
    ax_raster.axvspan(ARTIFACT_START_MS, ARTIFACT_END_MS, color="0.75", alpha=0.35, zorder=1)
    ax_raster.axvline(0, color="red", linestyle="--", linewidth=0.75, zorder=5)
    ax_raster.set_xlim(ARTIFACT_START_MS, NEXT_ARTIFACT_START_MS)
    ax_raster.set_ylim(-25, 525)
    ax_raster.set_xlabel("Time relative to pulse onset (ms)")
    ax_raster.set_ylabel("Stimulation pulse index")
    ax_rate.set_ylabel("Firing rate (spikes s$^{-1}$)")
    ax_rate.set_ylim(0, 320)
    ax_rate.legend(frameon=False, fontsize=6, ncol=1, loc="upper left")
    ax_raster.tick_params(width=0.6, length=2.5)
    ax_rate.tick_params(width=0.6, length=2.5)
    savefig(fig, PUB / "SF10a_pulse_cycle_minus0p4to9p6ms_curves_only_0to320")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    PUB = parser.parse_args().out
    PUB.mkdir(parents=True, exist_ok=True)
    main()
