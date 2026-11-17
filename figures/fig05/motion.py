"""Render side-by-side full-depth DREDge peak rasters with aligned drift traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors
from matplotlib.collections import LineCollection


FIGURE6_DIR = Path(__file__).resolve().parent
if str(FIGURE6_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE6_DIR))

from figure6_plot_style import SCALE_LABEL_PT, apply_figure6_style


FS = 30000.0
DRIFT_COLOR = "#1677a8"
RASTER_CMAP = "Greys"

apply_figure6_style()


def add_external_time_depth_scale(
    ax: plt.Axes,
    *,
    duration_s: float,
    depth_limits: tuple[float, float],
) -> None:
    y_min, y_max = depth_limits
    depth_span = y_max - y_min
    time_fraction = 60.0 / duration_s
    depth_fraction = 500.0 / depth_span
    x_depth = -0.095
    y_depth = -0.045
    x_time = -0.010
    y_time = -0.075
    ax.plot(
        [x_time, x_time + time_fraction],
        [y_time, y_time],
        color="black",
        lw=0.6,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        x_time + time_fraction / 2.0,
        y_time + 0.028,
        "60 s",
        ha="center",
        va="bottom",
        fontsize=SCALE_LABEL_PT,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.plot(
        [x_depth, x_depth],
        [y_depth, y_depth + depth_fraction],
        color="black",
        lw=0.6,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        x_depth - 0.018,
        y_depth + depth_fraction / 2.0,
        "500 um",
        ha="right",
        va="center",
        rotation=90,
        fontsize=SCALE_LABEL_PT,
        transform=ax.transAxes,
        clip_on=False,
    )


def plot_peak_raster(
    ax: plt.Axes,
    data: dict[str, np.ndarray | float | int],
    *,
    depth_limits: tuple[float, float],
    amplitude_vmax: float,
    event_height_um: float,
    event_linewidth: float,
    marker_alpha: float,
) -> None:
    times_s = np.asarray(data["times_s"], dtype=float)
    depth_um = np.asarray(data["depth_um"], dtype=float)
    amplitude_uv = np.asarray(data["amplitude_uv"], dtype=float)
    amp_norm = np.clip(amplitude_uv / max(amplitude_vmax, 1e-9), 0.0, 1.0)
    half_height = event_height_um / 2.0
    segments = np.stack(
        [
            np.column_stack([times_s, depth_um - half_height]),
            np.column_stack([times_s, depth_um + half_height]),
        ],
        axis=1,
    )
    line_colors = plt.get_cmap(RASTER_CMAP)(amp_norm)
    line_colors[:, 3] = marker_alpha
    ax.add_collection(LineCollection(
        segments,
        colors=line_colors,
        linewidths=event_linewidth,
        rasterized=True,
    ))
    y_min, y_max = depth_limits
    ax.set_xlim(0.0, float(data["duration_s"]))
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()


def plot_drift(
    ax: plt.Axes,
    data: dict[str, np.ndarray | float | int],
    *,
    drift_limit_um: float,
    show_ylabel: bool,
) -> None:
    times = np.asarray(data["motion_times_s"], dtype=float)
    displacement = np.asarray(data["displacement_um"], dtype=float)
    mean_drift = np.asarray(data["mean_drift_um"], dtype=float)
    ax.plot(
        times,
        displacement,
        color="#858585",
        alpha=0.32,
        lw=0.42,
        rasterized=False,
    )
    ax.plot(times, mean_drift, color=DRIFT_COLOR, lw=0.8)
    ax.axhline(0.0, color="#b5b5b5", lw=0.4, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(0.0, float(data["duration_s"]))
    ax.set_ylim(-drift_limit_um, drift_limit_um)
    ax.set_xlabel("Time (s)", labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Drift (um)", labelpad=1.5)
    else:
        ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.2)


def save_figure(fig: plt.Figure, output_stem: Path, formats: tuple[str, ...]) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        output = output_stem.with_suffix(f".{fmt}")
        fig.savefig(output, dpi=600 if fmt == "png" else 300, facecolor="white")
        written.append(output)
    return written


def render_composite(
    data_root: Path,
    output_stem: Path,
    *,
    formats: tuple[str, ...] = ("png",),
    event_height_um: float = 75.0,
    event_linewidth: float = 0.45,
    marker_alpha: float = 0.75,
    amplitude_vmax_percentile: float = 97.0,
    drift_limit_um: float | None = None,
) -> list[Path]:
    sessions = []
    for phase in (1, 2):
        with np.load(data_root/f"phase{phase}.npz") as data:
            sessions.append(dict(data))
    contact_min = min(float(session["contact_min_um"]) for session in sessions)
    contact_max = max(float(session["contact_max_um"]) for session in sessions)
    depth_pad_um = 30.0
    depth_limits = (contact_min - depth_pad_um, contact_max + depth_pad_um)

    all_amplitudes = np.concatenate([np.asarray(session["amplitude_uv"], dtype=float) for session in sessions])
    amplitude_vmax = float(np.nanpercentile(all_amplitudes, amplitude_vmax_percentile))

    if drift_limit_um is None:
        all_mean = np.concatenate([np.asarray(session["mean_drift_um"], dtype=float) for session in sessions])
        drift_limit_um = max(10.0, float(np.ceil(np.nanmax(np.abs(all_mean)) / 5.0) * 5.0))

    fig = plt.figure(figsize=(3.95, 2.05))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.3, 0.52],
        width_ratios=[1.0, 1.0],
        hspace=0.16,
        wspace=0.10,
        left=0.17,
        right=0.985,
        bottom=0.18,
        top=0.98,
    )
    raster_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    drift_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

    for i, session_data in enumerate(sessions):
        plot_peak_raster(
            raster_axes[i],
            session_data,
            depth_limits=depth_limits,
            amplitude_vmax=amplitude_vmax,
            event_height_um=event_height_um,
            event_linewidth=event_linewidth,
            marker_alpha=marker_alpha,
        )
        plot_drift(
            drift_axes[i],
            session_data,
            drift_limit_um=drift_limit_um,
            show_ylabel=i == 0,
        )
    add_external_time_depth_scale(
        raster_axes[0],
        duration_s=float(sessions[0]["duration_s"]),
        depth_limits=depth_limits,
    )

    norm = mcolors.Normalize(vmin=0.0, vmax=amplitude_vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(RASTER_CMAP))
    left_bbox = raster_axes[0].get_position()
    cax = fig.add_axes([
        left_bbox.x0 - 0.085,
        left_bbox.y0 + 0.16 * left_bbox.height,
        0.018,
        0.68 * left_bbox.height,
    ])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Amplitude (uV)", labelpad=2)
    cbar.ax.tick_params(pad=1.0)

    written = save_figure(fig, output_stem, formats)
    plt.close(fig)
    return written


