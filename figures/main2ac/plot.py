"""Reproduce Figure 2 panel assets.

This script consolidates the Figure 2 code that was previously spread across
notebooks and panel-specific snippets. It writes panel SVGs, and optionally
PNGs, into a local output directory.

Examples
--------
python make_figure2_panels.py --panel all
python make_figure2_panels.py --panel a c --png
python make_figure2_panels.py --output-dir outputs_figure2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyArrowPatch


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Figure2Paths:
    hanqing_folder: Path = Path("external/raw recording/from_hanqing")
    tracking_128_folder: Path = Path("external/EBL data/tracking_me/Processed_07-Mar-2024_153954")
    tracking_85_day1_folder: Path = Path("external/EBL data/tracking_me/Processed_30-Mar-2024_165548/waveforms_sorting_1")
    tracking_43_day1_folder: Path = Path("external/EBL data/tracking_me/Processed_19-Mar-2024_105200/waveforms_sorting_1")
    data_sorted_folder: Path = Path("external/data_sorted")

    @property
    def hanqing_dat(self) -> Path:
        return self.hanqing_folder / "hanqing.dat"

    @property
    def hanqing_probe(self) -> Path:
        return self.hanqing_folder / "EBL_ProbeInterface_hanqing.json"

    @property
    def hanqing_waveforms(self) -> Path:
        return self.hanqing_folder / "waveforms_curated_3"


CONNECTIONS_128 = [
    (4, 5), (5, 7), (14, 12), (15, 14), (18, 20), (27, 31), (29, 30),
    (30, 55), (54, 40), (54, 42), (54, 59), (59, 74), (65, 88),
    (82, 74), (84, 80), (88, 89), (89, 86), (89, 90), (91, 90),
    (93, 91), (93, 95),
]
CONNECTIONS_85 = [
    (4, 6), (6, 14), (11, 17), (13, 3), (13, 14), (14, 19), (19, 16),
    (19, 17), (20, 24), (22, 21), (22, 24), (26, 24), (26, 29),
    (32, 29), (40, 36), (45, 47),
]
CONNECTIONS_43 = [
    (3, 1), (3, 19), (11, 34), (15, 16), (25, 16), (25, 19),
    (26, 25), (27, 26), (34, 35),
]

CONTACT_MARKER_SIZE = 10
BUBBLE_SIZE_PER_UV = 2
BUBBLE_ALPHA_BC = 0.3
BUBBLE_ALPHA_D = 0.5
BUBBLE_EDGE_COLOR = "black"
BUBBLE_EDGE_WIDTH = 0.5
PANEL_D_CONTACT_MARKER_AREA_PT2 = 0.25
PANEL_D_PROBE_OUTLINE_WIDTH_PT = 0.3
PANEL_D_ARROW_WIDTH_PT = 0.8
PANEL_D_AXIS_LABEL_SIZE_PT = 7
PANEL_D_TICK_LABEL_SIZE_PT = 6
PANEL_D_FIGSIZE_IN = (2.5, 3.9)
PANEL_C_CONTACT_OFFSET_UV = 50.0
PANEL_C_CONTACT_X_MS = 1.0


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    png: bool = False,
    dpi: int = 300,
    *,
    tight: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    bbox_inches = "tight" if tight else None
    fig.savefig(svg_path, format="svg", bbox_inches=bbox_inches, transparent=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches=bbox_inches, transparent=True)
    if png:
        fig.savefig(
            output_dir / f"{stem}.png",
            format="png",
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=True,
        )
    plt.close(fig)
    print(f"saved {svg_path}")
    print(f"saved {pdf_path}")


def cache_file(cache_dir: Path, stem: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stem}.npz"


def load_npz_cache(path: Path, rebuild_cache: bool) -> dict[str, np.ndarray] | None:
    if rebuild_cache or not path.exists():
        return None
    print(f"loading cache: {path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz_cache(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"saved cache: {path}", flush=True)


def load_waveform_extractor(folder: Path, with_recording: bool | None = None):
    import spikeinterface.full as si

    require_path(folder, "waveform extractor folder")
    kwargs = {}
    if with_recording is not None:
        kwargs["with_recording"] = with_recording
    print(f"loading waveform extractor: {folder}", flush=True)
    return si.WaveformExtractor.load_from_folder(folder=folder, **kwargs)


def compute_unit_locations(we):
    from spikeinterface.postprocessing import compute_unit_locations as _compute_unit_locations

    return _compute_unit_locations(we)


def compute_quality_metrics(we):
    import spikeinterface.qualitymetrics as qm

    return qm.compute_quality_metrics(waveform_extractor=we, metric_names=["firing_rate", "snr"])


def get_template_extremum_channel(we):
    import spikeinterface.full as si

    return si.get_template_extremum_channel(we, outputs="index")


def load_direct_waveform_folder(folder: Path, require_unit_locations: bool = True) -> dict[str, np.ndarray]:
    """Read saved WaveformExtractor arrays without instantiating SpikeInterface."""
    require_path(folder, "waveform folder")
    recording_attributes_path = folder / "recording_info" / "recording_attributes.json"
    if recording_attributes_path.exists():
        with recording_attributes_path.open("r", encoding="utf-8") as f:
            recording_attributes = json.load(f)
        channel_ids = np.asarray(recording_attributes["channel_ids"])
        channel_locations = np.asarray(recording_attributes["properties"]["location"], dtype=float)
    else:
        with (folder / "recording.json").open("r", encoding="utf-8") as f:
            recording_attributes = json.load(f)
        channel_ids = np.asarray(
            recording_attributes.get("kwargs", {}).get("renamed_channel_ids")
            or recording_attributes.get("kwargs", {}).get("channel_ids")
        )
        channel_locations = np.asarray(recording_attributes["properties"]["location"], dtype=float)
    with (folder / "sorting.json").open("r", encoding="utf-8") as f:
        sorting_info = json.load(f)

    unit_ids = np.asarray(extract_unit_ids_from_sorting_json(sorting_info))
    templates = np.load(folder / "templates_average.npy", mmap_mode="r")
    unit_locations_path = folder / "unit_locations" / "unit_locations.npy"
    if unit_locations_path.exists():
        unit_locations = np.load(unit_locations_path)
    elif require_unit_locations:
        raise FileNotFoundError(f"Missing unit locations: {unit_locations_path}")
    else:
        unit_locations = np.empty((len(unit_ids), 0), dtype=float)
    return {
        "unit_ids": unit_ids,
        "channel_ids": channel_ids,
        "channel_locations": channel_locations,
        "templates": templates,
        "unit_locations": unit_locations,
    }


def extract_unit_ids_from_sorting_json(sorting_info: dict) -> list[int]:
    def walk(obj):
        if isinstance(obj, dict):
            kwargs = obj.get("kwargs")
            if isinstance(kwargs, dict):
                ids = kwargs.get("renamed_unit_ids") or kwargs.get("unit_ids")
                if ids is not None:
                    return ids
            for value in obj.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = walk(value)
                if found is not None:
                    return found
        return None

    unit_ids = walk(sorting_info)
    if unit_ids is None:
        raise KeyError("Could not find unit_ids or renamed_unit_ids in sorting.json")
    return unit_ids


def load_direct_quality_metrics(folder: Path) -> dict[int, tuple[float, float]]:
    metrics_path = folder / "quality_metrics" / "metrics.csv"
    if not metrics_path.exists():
        return {}
    out = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            unit_id = int(row[""])
            out[unit_id] = (float(row["firing_rate"]), float(row["snr"]))
    return out


def template_extremum_channel_indices(templates: np.ndarray) -> np.ndarray:
    p2p = np.ptp(np.asarray(templates), axis=1)
    return np.argmax(p2p, axis=1)


def make_unit_color_map(unit_ids: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    unit_ids = list(unit_ids)
    cmap = plt.colormaps.get_cmap("Set1")
    base_colors = [mcolors.to_rgba(color) for color in getattr(cmap, "colors", [cmap(i / 8) for i in range(9)])]
    return {int(unit_id): base_colors[i % len(base_colors)] for i, unit_id in enumerate(unit_ids)}


def get_panel_a_data(paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, "fig2a_raw_traces_data")
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    from scipy.signal import butter, sosfiltfilt

    require_path(paths.hanqing_dat, "Hanqing binary recording")

    sampling_rate = 30000
    num_channels = 128
    plot_start_s = 0.1
    plot_end_s = 0.2
    pad_s = 0.05
    read_start = int((plot_start_s - pad_s) * sampling_rate)
    read_end = int((plot_end_s + pad_s) * sampling_rate)
    crop_start = int(pad_s * sampling_rate)
    crop_end = crop_start + int((plot_end_s - plot_start_s) * sampling_rate)

    raw = np.memmap(paths.hanqing_dat, dtype="int16", mode="r")
    total_frames = raw.size // num_channels
    raw = raw[: total_frames * num_channels].reshape(total_frames, num_channels)
    segment = np.asarray(raw[read_start:read_end, :], dtype=float)

    sos = butter(3, [300, 6000], btype="bandpass", fs=sampling_rate, output="sos")
    filtered = sosfiltfilt(sos, segment, axis=0)[crop_start:crop_end]
    selected_indices = np.asarray(list(range(0, 63)) + [126, 127] + list(range(63, 126)), dtype=int)
    traces = filtered[:, selected_indices] / 5.0
    times = np.arange(traces.shape[0], dtype=float) / sampling_rate

    save_npz_cache(
        path,
        traces=traces.astype(np.float32),
        times=times,
        selected_indices=selected_indices,
        sampling_rate=np.asarray(sampling_rate),
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_a_raw_traces(
    paths: Figure2Paths, output_dir: Path, png: bool, cache_dir: Path, rebuild_cache: bool
) -> None:
    data = get_panel_a_data(paths, cache_dir, rebuild_cache)
    traces = data["traces"]
    times = data["times"]

    fig, ax = plt.subplots(figsize=(2.2, 10))
    spacing = np.nanpercentile(np.abs(traces), 99.5) * 2.0
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 200.0
    offsets = np.arange(traces.shape[1])[::-1] * spacing
    ax.plot(times, traces + offsets, color="black", lw=0.35)
    ax.axis("off")

    scalebar_x = times[0] + 0.005 * (times[-1] - times[0])
    scalebar_y = offsets.min() - spacing * 0.8
    ax.plot([scalebar_x, scalebar_x + 0.025], [scalebar_y, scalebar_y], "k", lw=2)
    ax.plot([scalebar_x, scalebar_x], [scalebar_y, scalebar_y + 1000], "k", lw=2)
    ax.text(scalebar_x + 0.0125, scalebar_y - spacing * 0.25, "25 ms",
            ha="center", va="top", fontsize=10)
    ax.text(scalebar_x - 0.002 * (times[-1] - times[0]), scalebar_y + 500, "1mV",
            ha="right", va="center", fontsize=10, rotation=90)
    ax.set_ylim(scalebar_y - spacing * 0.5, offsets.max() + spacing)
    ax.set_xlim(times[0], times[-1])

    save_figure(fig, output_dir, "fig2a_raw_traces", png=png)


def get_panel_b_data(paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, "fig2b_unit_locations_data")
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    direct = load_direct_waveform_folder(paths.hanqing_waveforms)
    all_channel_locations = direct["channel_locations"]
    unit_locations = direct["unit_locations"]
    unit_ids = direct["unit_ids"]
    amplitudes = np.max(np.abs(np.asarray(direct["templates"])), axis=(1, 2))

    save_npz_cache(
        path,
        channel_locations=all_channel_locations,
        unit_locations=unit_locations,
        unit_ids=np.asarray(unit_ids),
        amplitudes=np.asarray(amplitudes, dtype=float),
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_b_unit_locations(
    paths: Figure2Paths, output_dir: Path, png: bool, cache_dir: Path, rebuild_cache: bool
) -> None:
    data = get_panel_b_data(paths, cache_dir, rebuild_cache)
    all_channel_locations = data["channel_locations"]
    unit_locations = data["unit_locations"]
    unit_ids = data["unit_ids"]
    amplitudes = data["amplitudes"]

    fig, ax = plt.subplots(figsize=(2.2, 9.5))
    ax.set_xlim(all_channel_locations[:, 0].min() - 110, all_channel_locations[:, 0].max() + 110)
    ax.set_ylim(all_channel_locations[:, 1].min() - 80, all_channel_locations[:, 1].max() + 80)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter(
        all_channel_locations[:, 0],
        all_channel_locations[:, 1],
        color="black",
        s=CONTACT_MARKER_SIZE,
        edgecolors="none",
        linewidth=0,
        zorder=2,
    )

    colors = make_unit_color_map(unit_ids)
    for idx, unit_id in enumerate(unit_ids):
        mean_amplitude = amplitudes[idx]
        if not np.isfinite(mean_amplitude):
            continue
        x, y = unit_locations[idx, :2]
        ax.scatter(
            x,
            y,
            facecolors=[colors[int(unit_id)]],
            edgecolors=BUBBLE_EDGE_COLOR,
            s=mean_amplitude * BUBBLE_SIZE_PER_UV,
            zorder=3,
            alpha=BUBBLE_ALPHA_BC,
            linewidth=BUBBLE_EDGE_WIDTH,
            clip_on=False,
        )

    for amp in [50, 100, 150]:
        ax.scatter(
            [],
            [],
            facecolors="0.75",
            edgecolors=BUBBLE_EDGE_COLOR,
            alpha=BUBBLE_ALPHA_BC,
            linewidth=BUBBLE_EDGE_WIDTH,
            s=amp * BUBBLE_SIZE_PER_UV,
            label=f"{amp}uV",
        )
    ax.legend(
        scatterpoints=1,
        frameon=True,
        labelspacing=1.35,
        handletextpad=1.4,
        borderpad=0.85,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )

    save_figure(fig, output_dir, "fig2b_unit_locations", png=png)


def get_panel_c_data(
    paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool, selected_ch: int
) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, f"fig2c_waveform_distribution_idx{selected_ch}_data")
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    direct = load_direct_waveform_folder(paths.hanqing_waveforms)
    fs = 30000
    all_channel_locations = direct["channel_locations"]
    channel_ids = direct["channel_ids"]
    unit_ids = direct["unit_ids"]
    unit_locations = direct["unit_locations"]
    templates = np.asarray(direct["templates"])
    selected_channel_location = all_channel_locations[selected_ch]
    y_distances = np.abs(all_channel_locations[:, 1] - selected_channel_location[1])
    valid_y_range = (y_distances <= 100) & (all_channel_locations[:, 1] >= 0) & (all_channel_locations[:, 1] <= 2000)
    channel_indices = np.where(valid_y_range)[0]

    extremum_indices = template_extremum_channel_indices(templates)
    approved_unit_ids = [
        unit_id for unit_idx, unit_id in enumerate(unit_ids)
        if np.abs(all_channel_locations[extremum_indices[unit_idx], 1] - selected_channel_location[1]) <= 100
    ]

    mean_waveforms = []
    approved_unit_locations = []
    approved_amplitudes = []
    unit_id_to_index = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
    for unit_id in approved_unit_ids:
        unit_index = unit_id_to_index[unit_id]
        mean_waveforms.append(np.asarray(templates[unit_index])[:, channel_indices].T)
        approved_unit_locations.append(unit_locations[unit_index, :2])
        approved_amplitudes.append(np.max(np.abs(templates[unit_index])))

    if mean_waveforms:
        mean_waveforms_array = np.stack(mean_waveforms, axis=0)
    else:
        mean_waveforms_array = np.empty((0, len(channel_indices), 0), dtype=float)

    save_npz_cache(
        path,
        fs=np.asarray(fs),
        selected_ch_index=np.asarray(selected_ch),
        selected_ch_id=np.asarray(channel_ids[selected_ch]),
        channel_locations=all_channel_locations,
        selected_channel_location=selected_channel_location,
        channel_indices=channel_indices,
        all_unit_ids=np.asarray(unit_ids),
        approved_unit_ids=np.asarray(approved_unit_ids),
        approved_unit_locations=np.asarray(approved_unit_locations, dtype=float),
        approved_amplitudes=np.asarray(approved_amplitudes, dtype=float),
        mean_waveforms=mean_waveforms_array,
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_c_waveform_distribution(
    paths: Figure2Paths,
    output_dir: Path,
    png: bool,
    cache_dir: Path,
    rebuild_cache: bool,
    selected_ch: int | Iterable[int] = (86, 119),
    amplitude_scale: float = 1.2,
    waveform_box_y_scale: float = 3.0,
    waveform_ylim: tuple[float, float] = (-320.0, 160.0),
) -> None:
    if isinstance(selected_ch, int):
        selected_channels = [selected_ch]
    else:
        selected_channels = list(selected_ch)

    panel_data = [get_panel_c_data(paths, cache_dir, rebuild_cache, ch) for ch in selected_channels]
    fs = int(panel_data[0]["fs"])
    all_channel_locations = panel_data[0]["channel_locations"]
    all_unit_ids = panel_data[0]["all_unit_ids"]
    unit_colors = make_unit_color_map(all_unit_ids)

    if len(panel_data) == 1:
        fig = plt.figure(figsize=(4.2, 9 / 2))
        fig.subplots_adjust(left=0.25, right=0.62)
        overview_rect = [0.88, 0.1, 0.07, 0.8]
    else:
        fig = plt.figure(figsize=((3.0 * len(panel_data) + 0.9) / 2, 9 / 2))
        overview_rect = [0.92, 0.1, 0.08, 0.8]
    outer_grid = GridSpec(1, len(panel_data), figure=fig, wspace=0.28, hspace=0.1)
    section_colors = sns.color_palette("Set1", max(3, len(panel_data)))
    sample_waveform_ax = None

    for section_idx, data in enumerate(panel_data):
        selected_channel_location = data["selected_channel_location"]
        channel_indices = data["channel_indices"]
        approved_unit_ids = data["approved_unit_ids"]
        approved_unit_locations = data["approved_unit_locations"]
        approved_amplitudes = data["approved_amplitudes"]
        mean_waveforms = data["mean_waveforms"]
        selected_ch_id = data["selected_ch_id"].item()
        selected_ch_index = int(data["selected_ch_index"])

        inner_grid = GridSpecFromSubplotSpec(
            9, 3, wspace=0.1, hspace=0.1, subplot_spec=outer_grid[section_idx]
        )

        if len(approved_unit_ids) == 0:
            print(f"warning: no units found within 100 um of channel index {selected_ch_index}")

        color_dict = {int(unit_id): unit_colors[int(unit_id)] for unit_id in approved_unit_ids}

        x_positions = all_channel_locations[channel_indices, 0]
        y_positions = all_channel_locations[channel_indices, 1]
        x_min, x_max = x_positions.min(), x_positions.max()
        y_min, y_max = y_positions.min(), y_positions.max()
        x_span = max(float(x_max - x_min), 1.0)
        y_span = max(float(y_max - y_min), 1.0)

        axes_by_slot = {}
        bottom_unit_amplitudes = []
        for unit_pos, unit_id in enumerate(approved_unit_ids):
            for local_ch_pos, ch_idx in enumerate(channel_indices):
                ch_location = all_channel_locations[ch_idx]
                x_norm = (ch_location[0] - x_min) / x_span
                y_norm = (ch_location[1] - y_min) / y_span
                x_idx = int(np.round(x_norm * 2))
                y_idx = 8 - int(np.round(y_norm * 8))
                ax = axes_by_slot.get((y_idx, x_idx))
                if ax is None:
                    ax = fig.add_subplot(inner_grid[y_idx, x_idx])
                    ax.axis("off")
                    ax.set_ylim(*waveform_ylim)
                    ax.set_xlim(0, 3)
                    scale_axes_box_from_center(ax, y_scale=waveform_box_y_scale)
                    ax.scatter(
                        PANEL_C_CONTACT_X_MS,
                        PANEL_C_CONTACT_OFFSET_UV,
                        color="black",
                        s=CONTACT_MARKER_SIZE,
                        edgecolors="none",
                        linewidth=0,
                        zorder=20,
                        clip_on=False,
                    )
                    axes_by_slot[(y_idx, x_idx)] = ax
                    if sample_waveform_ax is None:
                        sample_waveform_ax = ax
                mean_waveform = mean_waveforms[unit_pos, local_ch_pos] * amplitude_scale
                times = np.arange(mean_waveform.shape[0]) / fs * 1000
                ax.plot(times, mean_waveform, color=color_dict[int(unit_id)], lw=0.3, alpha=0.95)
                if y_idx >= 6:
                    raw_waveform = mean_waveforms[unit_pos, local_ch_pos]
                    bottom_unit_amplitudes.append(
                        (
                            float(np.max(np.abs(raw_waveform))),
                            float(np.ptp(raw_waveform)),
                            int(unit_id),
                            int(ch_idx),
                            int(y_idx),
                            int(x_idx),
                        )
                    )

        if bottom_unit_amplitudes:
            print(f"panel c bottom-row largest amplitudes for selected channel index {selected_ch_index} (id {selected_ch_id}):")
            for abs_amp, p2p_amp, unit_id, ch_idx, y_idx, x_idx in sorted(bottom_unit_amplitudes, reverse=True)[:12]:
                print(
                    f"  unit {unit_id}, channel_index {ch_idx}, grid=({y_idx},{x_idx}), "
                    f"max_abs={abs_amp:.1f} uV, p2p={p2p_amp:.1f} uV"
                )

        title_ax = fig.add_subplot(outer_grid[section_idx], frameon=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.set_title(f"Ch {selected_ch_id}", fontsize=9, pad=4)

        bubble_ax = fig.add_subplot(outer_grid[section_idx], frameon=False)
        bubble_ax.set_xlim(x_min, x_max)
        bubble_ax.set_ylim(y_min, y_max)
        expand_overlay_axes_with_data_padding(bubble_ax, x_pad_fraction=0.45)
        bubble_ax.set_xticks([])
        bubble_ax.set_yticks([])
        bubble_ax.patch.set_alpha(0)
        bubble_ax.set_zorder(10)
        bubble_y_offset = (
            PANEL_C_CONTACT_OFFSET_UV
            / max(float(waveform_ylim[1] - waveform_ylim[0]), 1.0)
            * y_span
        )
        bubble_x_offset = (
            (PANEL_C_CONTACT_X_MS - 1.5)
            / max(float(3.0 - 0.0), 1.0)
            * x_span
        )
        for idx, unit_id in enumerate(approved_unit_ids):
            x, y = approved_unit_locations[idx, :2]
            bubble_ax.scatter(
                x + bubble_x_offset,
                y + bubble_y_offset,
                facecolors=[color_dict[int(unit_id)]],
                edgecolors=BUBBLE_EDGE_COLOR,
                s=approved_amplitudes[idx] * BUBBLE_SIZE_PER_UV,
                alpha=BUBBLE_ALPHA_BC,
                linewidths=BUBBLE_EDGE_WIDTH,
                clip_on=False,
                zorder=10,
            )

    overview_ax = fig.add_axes(overview_rect)
    overview_ax.scatter(
        all_channel_locations[:, 0],
        all_channel_locations[:, 1],
        c="black",
        s=CONTACT_MARKER_SIZE,
        edgecolors="none",
        linewidth=0,
    )
    for section_idx, data in enumerate(panel_data):
        selected_channel_location = data["selected_channel_location"]
        selected_ch_id = data["selected_ch_id"].item()
        color = section_colors[section_idx]
        overview_ax.scatter(
            selected_channel_location[0], selected_channel_location[1],
            c=[color], s=35, label=f"Ch {selected_ch_id}"
        )
        overview_ax.add_patch(Circle(selected_channel_location, radius=100, color=color, fill=False, linestyle="--"))
    overview_ax.set_aspect("equal")
    overview_ax.axis("off")

    if sample_waveform_ax is not None:
        add_panel_c_scale_bar(fig, sample_waveform_ax, amplitude_scale)

    channel_suffix = "_".join(f"ch{data['selected_ch_id'].item()}" for data in panel_data)
    save_figure(fig, output_dir, f"fig2c_waveform_distribution_{channel_suffix}", png=png)
    panel_c_waveform_distribution_debug(
        panel_data,
        all_channel_locations,
        fs,
        output_dir,
        png,
        amplitude_scale,
        channel_suffix,
        waveform_box_y_scale,
        waveform_ylim,
    )


def scale_axes_box_from_center(ax: plt.Axes, x_scale: float = 1.0, y_scale: float = 1.0) -> None:
    """Resize an axes box around its center without changing its data coordinates."""
    bbox = ax.get_position()
    center_x = bbox.x0 + bbox.width / 2
    center_y = bbox.y0 + bbox.height / 2
    width = bbox.width * x_scale
    height = bbox.height * y_scale
    ax.set_position([center_x - width / 2, center_y - height / 2, width, height])


def expand_overlay_axes_with_data_padding(ax: plt.Axes, x_pad_fraction: float = 0.0, y_pad_fraction: float = 0.0) -> None:
    """Expand an overlay axes and its data limits while keeping old data positions fixed."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_position()
    width = bbox.width
    height = bbox.height

    new_x0 = bbox.x0 - width * x_pad_fraction
    new_width = width * (1 + 2 * x_pad_fraction)
    new_y0 = bbox.y0 - height * y_pad_fraction
    new_height = height * (1 + 2 * y_pad_fraction)
    ax.set_position([new_x0, new_y0, new_width, new_height])

    x_pad = (x1 - x0) * x_pad_fraction
    y_pad = (y1 - y0) * y_pad_fraction
    ax.set_xlim(x0 - x_pad, x1 + x_pad)
    ax.set_ylim(y0 - y_pad, y1 + y_pad)


def add_panel_c_scale_bar(fig: plt.Figure, reference_ax: plt.Axes, amplitude_scale: float) -> None:
    """Draw a scale bar in figure coordinates using a waveform axis' data scale."""
    bbox = reference_ax.get_position()
    xlim = reference_ax.get_xlim()
    ylim = reference_ax.get_ylim()
    scale_time_ms = 1.0
    scale_voltage_uv = 100.0
    bar_width = bbox.width * scale_time_ms / (xlim[1] - xlim[0])
    bar_height = bbox.height * (scale_voltage_uv * amplitude_scale) / (ylim[1] - ylim[0])
    x0 = 0.80
    y0 = 0.055

    fig.add_artist(mlines.Line2D([x0, x0 + bar_width], [y0, y0], transform=fig.transFigure, color="black", lw=1.2))
    fig.add_artist(mlines.Line2D([x0, x0], [y0, y0 + bar_height], transform=fig.transFigure, color="black", lw=1.2))
    fig.text(x0 + bar_width / 2, y0 - 0.012, "1 ms", ha="center", va="top", fontsize=6)
    fig.text(x0 - 0.012, y0 + bar_height / 2, "100 uV", ha="right", va="center", rotation=90, fontsize=6)


def panel_c_waveform_distribution_debug(
    panel_data: list[dict[str, np.ndarray]],
    all_channel_locations: np.ndarray,
    fs: int,
    output_dir: Path,
    png: bool,
    amplitude_scale: float,
    channel_suffix: str,
    waveform_box_y_scale: float,
    waveform_ylim: tuple[float, float],
) -> None:
    if len(panel_data) == 1:
        fig = plt.figure(figsize=(5.8, 10.8))
        fig.subplots_adjust(left=0.22, right=0.66)
        overview_rect = [0.90, 0.23, 0.045, 0.64]
    else:
        fig = plt.figure(figsize=(3.8 * len(panel_data) + 1.4, 10.8))
        overview_rect = [0.93, 0.23, 0.055, 0.64]
    outer_grid = GridSpec(2, len(panel_data), figure=fig, height_ratios=[9.0, 1.8], wspace=0.34, hspace=0.08)
    section_colors = sns.color_palette("Set1", max(3, len(panel_data)))
    all_unit_ids = panel_data[0]["all_unit_ids"]
    unit_colors = make_unit_color_map(all_unit_ids)

    for section_idx, data in enumerate(panel_data):
        selected_channel_location = data["selected_channel_location"]
        channel_indices = data["channel_indices"]
        approved_unit_ids = data["approved_unit_ids"]
        approved_unit_locations = data["approved_unit_locations"]
        approved_amplitudes = data["approved_amplitudes"]
        mean_waveforms = data["mean_waveforms"]
        selected_ch_id = data["selected_ch_id"].item()
        selected_ch_index = int(data["selected_ch_index"])

        inner_grid = GridSpecFromSubplotSpec(
            9, 3, wspace=0.1, hspace=0.1, subplot_spec=outer_grid[0, section_idx]
        )
        color_dict = {int(unit_id): unit_colors[int(unit_id)] for unit_id in approved_unit_ids}

        x_positions = all_channel_locations[channel_indices, 0]
        y_positions = all_channel_locations[channel_indices, 1]
        x_min, x_max = x_positions.min(), x_positions.max()
        y_min, y_max = y_positions.min(), y_positions.max()
        x_span = max(float(x_max - x_min), 1.0)
        y_span = max(float(y_max - y_min), 1.0)

        axes_by_slot = {}
        bottom_unit_amplitudes = []
        channel_slot_labels = {}
        for unit_pos, unit_id in enumerate(approved_unit_ids):
            for local_ch_pos, ch_idx in enumerate(channel_indices):
                ch_location = all_channel_locations[ch_idx]
                x_norm = (ch_location[0] - x_min) / x_span
                y_norm = (ch_location[1] - y_min) / y_span
                x_idx = int(np.round(x_norm * 2))
                y_idx = 8 - int(np.round(y_norm * 8))
                ax = axes_by_slot.get((y_idx, x_idx))
                if ax is None:
                    ax = fig.add_subplot(inner_grid[y_idx, x_idx])
                    ax.axis("off")
                    ax.set_ylim(*waveform_ylim)
                    ax.set_xlim(0, 3)
                    scale_axes_box_from_center(ax, y_scale=waveform_box_y_scale)
                    ax.scatter(
                        PANEL_C_CONTACT_X_MS,
                        PANEL_C_CONTACT_OFFSET_UV,
                        color="black",
                        s=CONTACT_MARKER_SIZE,
                        edgecolors="none",
                        linewidth=0,
                        zorder=20,
                        clip_on=False,
                    )
                    axes_by_slot[(y_idx, x_idx)] = ax
                    channel_slot_labels[(y_idx, x_idx)] = ch_idx
                mean_waveform = mean_waveforms[unit_pos, local_ch_pos] * amplitude_scale
                times = np.arange(mean_waveform.shape[0]) / fs * 1000
                ax.plot(times, mean_waveform, color=color_dict[int(unit_id)], lw=0.3, alpha=0.95)
                if y_idx >= 6:
                    raw_waveform = mean_waveforms[unit_pos, local_ch_pos]
                    bottom_unit_amplitudes.append(
                        (
                            float(np.max(np.abs(raw_waveform))),
                            float(np.ptp(raw_waveform)),
                            int(unit_id),
                            int(ch_idx),
                            int(y_idx),
                            int(x_idx),
                            color_dict[int(unit_id)],
                        )
                    )

        for slot, ax in axes_by_slot.items():
            ch_idx = channel_slot_labels[slot]
            ax.text(
                0.02,
                0.95,
                f"ch {ch_idx}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                color="0.25",
            )

        title_ax = fig.add_subplot(outer_grid[0, section_idx], frameon=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.set_title(f"Section: channel index {selected_ch_index} / id {selected_ch_id}", fontsize=9, pad=4)

        bubble_ax = fig.add_subplot(outer_grid[0, section_idx], frameon=False)
        bubble_ax.set_xlim(x_min, x_max)
        bubble_ax.set_ylim(y_min, y_max)
        expand_overlay_axes_with_data_padding(bubble_ax, x_pad_fraction=0.45)
        bubble_ax.set_xticks([])
        bubble_ax.set_yticks([])
        bubble_ax.patch.set_alpha(0)
        bubble_ax.set_zorder(10)
        bubble_y_offset = (
            PANEL_C_CONTACT_OFFSET_UV
            / max(float(waveform_ylim[1] - waveform_ylim[0]), 1.0)
            * y_span
        )
        bubble_x_offset = (
            (PANEL_C_CONTACT_X_MS - 1.5)
            / max(float(3.0 - 0.0), 1.0)
            * x_span
        )
        for idx, unit_id in enumerate(approved_unit_ids):
            x, y = approved_unit_locations[idx, :2]
            bubble_ax.scatter(
                x + bubble_x_offset,
                y + bubble_y_offset,
                facecolors=[color_dict[int(unit_id)]],
                edgecolors=BUBBLE_EDGE_COLOR,
                s=approved_amplitudes[idx] * BUBBLE_SIZE_PER_UV,
                alpha=BUBBLE_ALPHA_BC,
                linewidths=BUBBLE_EDGE_WIDTH,
                clip_on=False,
                zorder=10,
            )

        legend_ax = fig.add_subplot(outer_grid[1, section_idx])
        legend_ax.axis("off")
        legend_ax.text(0.0, 1.0, "Unit color key", fontsize=7, ha="left", va="top", weight="bold")
        for i, unit_id in enumerate(approved_unit_ids):
            col = i % 4
            row = i // 4
            x0 = 0.02 + col * 0.24
            y0 = 0.82 - row * 0.12
            if y0 < 0.05:
                break
            legend_ax.plot([x0, x0 + 0.05], [y0, y0], color=color_dict[int(unit_id)], lw=2)
            legend_ax.text(x0 + 0.06, y0, str(unit_id), color=color_dict[int(unit_id)], fontsize=6, va="center")

        amp_ax = legend_ax.inset_axes([0.0, -0.52, 1.0, 0.48])
        amp_ax.axis("off")
        amp_ax.text(0.0, 1.0, "Largest bottom-row amplitudes", fontsize=7, ha="left", va="top", weight="bold")
        for i, (abs_amp, p2p_amp, unit_id, ch_idx, y_idx, x_idx, color) in enumerate(
            sorted(bottom_unit_amplitudes, reverse=True)[:8]
        ):
            y0 = 0.78 - i * 0.11
            amp_ax.text(
                0.0,
                y0,
                f"unit {unit_id} | ch {ch_idx} | grid {y_idx},{x_idx} | abs {abs_amp:.1f} uV | p2p {p2p_amp:.1f} uV",
                color=color,
                fontsize=5.5,
                ha="left",
                va="center",
            )

    overview_ax = fig.add_axes(overview_rect)
    overview_ax.scatter(
        all_channel_locations[:, 0],
        all_channel_locations[:, 1],
        c="black",
        s=CONTACT_MARKER_SIZE,
        edgecolors="none",
        linewidth=0,
    )
    for section_idx, data in enumerate(panel_data):
        selected_channel_location = data["selected_channel_location"]
        selected_ch_id = data["selected_ch_id"].item()
        color = section_colors[section_idx]
        overview_ax.scatter(selected_channel_location[0], selected_channel_location[1], c=[color], s=35)
        overview_ax.text(
            selected_channel_location[0] + 18,
            selected_channel_location[1],
            f"{selected_ch_id}",
            color=color,
            fontsize=6,
            va="center",
        )
        overview_ax.add_patch(Circle(selected_channel_location, radius=100, color=color, fill=False, linestyle="--"))
    overview_ax.set_aspect("equal")
    overview_ax.axis("off")

    save_figure(fig, output_dir, f"fig2c_waveform_distribution_{channel_suffix}_debug", png=png)


def draw_connections(ax, connections, unit_locations, unit_id_to_index):
    for start_unit_id, end_unit_id in connections:
        if start_unit_id not in unit_id_to_index or end_unit_id not in unit_id_to_index:
            continue
        start_location = unit_locations[unit_id_to_index[start_unit_id]]
        end_location = unit_locations[unit_id_to_index[end_unit_id]]
        if start_location[1] <= 1000 or end_location[1] <= 1000:
            continue
        arrow = FancyArrowPatch(
            posA=start_location,
            posB=end_location,
            arrowstyle="->",
            color="black",
            linewidth=PANEL_D_ARROW_WIDTH_PT,
            connectionstyle="arc3,rad=0.2",
            mutation_scale=10,
            alpha=1,
            clip_on=False,
        )
        ax.add_patch(arrow)


def get_panel_d_layout_data(folder: Path, cache_dir: Path, rebuild_cache: bool, label: str) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, f"fig2d_layout_{label}_data")
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    direct = load_direct_waveform_folder(folder)
    all_channel_locations = direct["channel_locations"]
    unit_locations = direct["unit_locations"]
    templates = np.asarray(direct["templates"])
    unit_ids = np.asarray(direct["unit_ids"])
    unit_metrics = load_direct_quality_metrics(folder)
    amplitudes = np.max(np.abs(templates), axis=(1, 2))
    valid_mask = []

    for idx, unit_id in enumerate(unit_ids):
        mean_amplitude = float(amplitudes[idx])
        x, y = unit_locations[idx, :2]
        firing_rate, snr = unit_metrics.get(int(unit_id), (np.inf, np.inf))
        is_valid = (
            firing_rate >= 0.1
            and snr >= 3.5
            and y >= 1000
        )
        valid_mask.append(is_valid)

    save_npz_cache(
        path,
        channel_locations=all_channel_locations,
        unit_locations=unit_locations,
        unit_ids=unit_ids,
        amplitudes=np.asarray(amplitudes, dtype=float),
        valid_mask=np.asarray(valid_mask, dtype=bool),
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_d_layout_connectivity(
    paths: Figure2Paths, output_dir: Path, png: bool, cache_dir: Path, rebuild_cache: bool
) -> None:
    layout_data = [
        get_panel_d_layout_data(paths.tracking_128_folder / "waveforms_sorting_1", cache_dir, rebuild_cache, "128"),
        get_panel_d_layout_data(paths.tracking_85_day1_folder, cache_dir, rebuild_cache, "85"),
        get_panel_d_layout_data(paths.tracking_43_day1_folder, cache_dir, rebuild_cache, "43"),
    ]

    connection_sets = [CONNECTIONS_128, CONNECTIONS_85, CONNECTIONS_43]
    titles = ["128ch", "85ch", "43ch"]

    fig, axes = plt.subplots(1, 3, figsize=PANEL_D_FIGSIZE_IN)
    plt.subplots_adjust(left=0.25, right=0.68, bottom=0.11, top=0.94, wspace=0.55)
    xlim = (-50, 50)
    ylim = (1000, 2000)

    all_unit_ids = sorted(set().union(*(set(data["unit_ids"].tolist()) for data in layout_data)))
    palette = sns.color_palette("deep", len(all_unit_ids))
    color_dict = {unit_id: mcolors.to_rgba_array(palette)[i] for i, unit_id in enumerate(all_unit_ids)}

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(titles[ax_idx], fontsize=PANEL_D_AXIS_LABEL_SIZE_PT, pad=2)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.set_xticks([-40, 0, 40])
        ax.set_xticklabels([-40, 0, 40])
        ax.tick_params(
            axis="both",
            labelsize=PANEL_D_TICK_LABEL_SIZE_PT,
            width=0.5,
            length=2,
            pad=1,
        )
        if ax_idx == 0:
            ax.set_ylabel("Distance(um)", fontsize=PANEL_D_AXIS_LABEL_SIZE_PT, labelpad=2)
        else:
            ax.get_yaxis().set_visible(False)

        data = layout_data[ax_idx]
        unit_ids = data["unit_ids"]
        unit_id_to_index = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
        all_channel_locations = data["channel_locations"]
        unit_locations = data["unit_locations"]
        amplitudes = data["amplitudes"]
        valid_mask = data["valid_mask"]

        draw_connections(ax, connection_sets[ax_idx], unit_locations, unit_id_to_index)

        ax.scatter(
            all_channel_locations[:, 0],
            all_channel_locations[:, 1],
            color="black",
            s=PANEL_D_CONTACT_MARKER_AREA_PT2,
            zorder=2,
            edgecolors="none",
            linewidth=0,
        )

        for idx, unit_id in enumerate(unit_ids):
            mean_amplitude = amplitudes[idx]
            x, y = unit_locations[idx, :2]
            if valid_mask[idx] and np.isfinite(mean_amplitude):
                ax.scatter(
                    x,
                    y,
                    facecolors="lightgray",
                    edgecolors=BUBBLE_EDGE_COLOR,
                    s=mean_amplitude * BUBBLE_SIZE_PER_UV,
                    zorder=3,
                    alpha=BUBBLE_ALPHA_D,
                    linewidth=BUBBLE_EDGE_WIDTH,
                )

        ax.add_patch(
            plt.Rectangle(
                (-48, 1000),
                96,
                1000,
                linewidth=PANEL_D_PROBE_OUTLINE_WIDTH_PT,
                edgecolor="black",
                facecolor="none",
            )
        )

    for amp in [50, 100, 150]:
        axes[-1].scatter(
            [],
            [],
            facecolors="lightgray",
            edgecolors=BUBBLE_EDGE_COLOR,
            alpha=BUBBLE_ALPHA_D,
            linewidth=BUBBLE_EDGE_WIDTH,
            s=amp * BUBBLE_SIZE_PER_UV,
            label=f"{amp}uV",
        )
    axes[-1].legend(
        scatterpoints=1,
        frameon=True,
        labelspacing=1.35,
        handletextpad=0.5,
        borderpad=0.3,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
    )
    axes[1].set_xlabel("Distance(um)", fontsize=PANEL_D_AXIS_LABEL_SIZE_PT, labelpad=2)

    save_figure(fig, output_dir, "fig2d_layout_connectivity", png=png, tight=False)


def layout_folder(paths: Figure2Paths, layout: int, day: int) -> Path:
    if layout == 128:
        return paths.tracking_128_folder / f"waveforms_sorting_{day}"
    return paths.tracking_128_folder / f"waveforms_layout_{layout}_{day}"


def panel_e_sorting_folders(paths: Figure2Paths) -> list[tuple[int, int, Path]]:
    raise RuntimeError("panel_e_sorting_folders is deprecated; use panel_e_animal_layout_folders")


def panel_e_animal_layout_folders(paths: Figure2Paths) -> list[tuple[str, int, Path]]:
    """Verified independent animal-level sortings for panel e."""
    root = paths.data_sorted_folder
    density_root = root / "tracking_vs_density"
    return [
        ("EBL2", 128, root / "EBL2_Oct11" / "Processed_19-Mar-2024_150636" / "we_curated_2_new_isi_1"),
        ("EBL6", 128, root / "EBL6_Feb16" / "Processed_20-Mar-2024_201615" / "we_curated_2_new_isi_1"),
        ("EBL7", 128, root / "EBL7_Mar02" / "Processed_20-Mar-2024_202400" / "we_curated_2_new_isi_1"),
        ("EBL8", 43, density_root / "EBL8" / "Processed_05-Jun-2024_195943" / "we_curated_2_new_isi_1"),
        ("EBL8", 85, density_root / "EBL8" / "Processed_05-Jun-2024_200323" / "we_curated_2_new_isi_1"),
        ("EBL8", 128, root / "EBL8-Apr8" / "Processed_08-Apr-2024_130918" / "we_curated_2_new_isi_1"),
        ("EBL9", 43, density_root / "EBL9" / "Processed_05-Jun-2024_200441" / "we_curated_2_new_isi_1"),
        ("EBL9", 85, density_root / "EBL9" / "Processed_05-Jun-2024_200620" / "we_curated_2_new_isi_1"),
        ("EBL9", 128, root / "EBL9-Apr8" / "Processed_08-Apr-2024_131345" / "we_curated_2_new_isi_1"),
    ]


def get_panel_e_data(paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, "fig2e_density_summary_animal_layout_data")
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    layouts = [43, 85, 128]
    animals = ["EBL2", "EBL6", "EBL7", "EBL8", "EBL9"]
    layout_to_index = {layout: idx for idx, layout in enumerate(layouts)}
    animal_to_index = {animal: idx for idx, animal in enumerate(animals)}
    yield_values = np.full((len(layouts), len(animals)), np.nan, dtype=float)
    localized_values = np.full((len(layouts), len(animals)), np.nan, dtype=float)

    for animal, layout, folder in panel_e_animal_layout_folders(paths):
        if not folder.exists():
            print(f"warning: skipping missing sorting folder: {folder}")
            continue
        layout_idx = layout_to_index[layout]
        animal_idx = animal_to_index[animal]
        direct = load_direct_waveform_folder(folder, require_unit_locations=False)
        unit_ids = list(direct["unit_ids"])
        templates = np.asarray(direct["templates"])
        yield_values[layout_idx, animal_idx] = len(unit_ids)
        all_channel_locations = direct["channel_locations"]
        extremum_indices = template_extremum_channel_indices(templates)

        per_unit_counts = []
        for unit_idx, unit_id in enumerate(unit_ids):
            extremum_ch = extremum_indices[unit_idx]
            extremum_location = all_channel_locations[extremum_ch]
            y_distances = np.abs(all_channel_locations[:, 1] - extremum_location[1])
            valid_y_range = (
                (y_distances <= 200)
                & (all_channel_locations[:, 1] >= 0)
                & (all_channel_locations[:, 1] <= 2000)
            )
            channel_indices = np.where(valid_y_range)[0]
            template = np.asarray(templates[unit_idx])[:, channel_indices]
            valid_p2p = np.ptp(template, axis=0)
            if valid_p2p.size == 0 or np.max(valid_p2p) == 0:
                continue
            per_unit_counts.append(int(np.sum(valid_p2p >= 0.2 * np.max(valid_p2p))))

        if per_unit_counts:
            localized_values[layout_idx, animal_idx] = float(np.mean(per_unit_counts))

        print(
            f"{animal} layout {layout} sorting: yield={yield_values[layout_idx, animal_idx]:.0f}, "
            f"localized={localized_values[layout_idx, animal_idx]:.2f}"
        )

    save_npz_cache(
        path,
        layouts=np.asarray(layouts, dtype=float),
        animals=np.asarray(animals, dtype="U8"),
        yield_values=yield_values,
        localized_values=localized_values,
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_e_localized_layout_folder(paths: Figure2Paths, layout: int, day: int) -> Path:
    if layout == 128:
        return paths.tracking_128_folder / f"waveforms_sorting_{day}"
    return paths.tracking_128_folder / f"waveforms_layout_{layout}_{day}"


def get_panel_e_localized_layout_data(
    paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool
) -> dict[str, np.ndarray]:
    return get_panel_e_localized_layout_data_for_layouts(
        paths=paths,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        layouts=[43, 85, 128],
        cache_stem="fig2e_localized_electrodes_layout_day2_43_85_128_data",
    )


def get_panel_e_localized_layout_data_all5(
    paths: Figure2Paths, cache_dir: Path, rebuild_cache: bool
) -> dict[str, np.ndarray]:
    return get_panel_e_localized_layout_data_for_layouts(
        paths=paths,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        layouts=[43, 64, 85, 106, 128],
        cache_stem="fig2e_localized_electrodes_layout_day2_all5_data",
    )


def get_panel_e_localized_layout_data_for_layouts(
    paths: Figure2Paths,
    cache_dir: Path,
    rebuild_cache: bool,
    layouts: list[int],
    cache_stem: str,
) -> dict[str, np.ndarray]:
    path = cache_file(cache_dir, cache_stem)
    cached = load_npz_cache(path, rebuild_cache)
    if cached is not None:
        return cached

    days = [2]
    counts_by_layout = []
    means = []
    stds = []

    for layout in layouts:
        layout_counts = []
        for day in days:
            folder = panel_e_localized_layout_folder(paths, layout, day)
            direct = load_direct_waveform_folder(folder, require_unit_locations=False)
            templates = np.asarray(direct["templates"])
            channel_locations = direct["channel_locations"]
            extremum_indices = template_extremum_channel_indices(templates)

            for unit_idx in range(len(direct["unit_ids"])):
                extremum_ch = extremum_indices[unit_idx]
                extremum_location = channel_locations[extremum_ch]
                y_distances = np.abs(channel_locations[:, 1] - extremum_location[1])
                valid_y_range = (
                    (y_distances <= 200)
                    & (channel_locations[:, 1] >= 0)
                    & (channel_locations[:, 1] <= 2000)
                )
                channel_indices = np.where(valid_y_range)[0]
                template = templates[unit_idx][:, channel_indices]
                valid_p2p = np.ptp(template, axis=0)
                if valid_p2p.size == 0 or np.max(valid_p2p) == 0:
                    continue
                layout_counts.append(int(np.sum(valid_p2p >= 0.2 * np.max(valid_p2p))))

        counts = np.asarray(layout_counts, dtype=float)
        counts_by_layout.append(counts)
        means.append(float(np.mean(counts)))
        stds.append(float(np.std(counts)))
        print(
            f"localized layout {layout} day 2: n={counts.size}, "
            f"mean={means[-1]:.2f}, std={stds[-1]:.2f}"
        )

    max_count = max(counts.size for counts in counts_by_layout)
    count_matrix = np.full((len(layouts), max_count), np.nan, dtype=float)
    for idx, counts in enumerate(counts_by_layout):
        count_matrix[idx, : counts.size] = counts

    save_npz_cache(
        path,
        layouts=np.asarray(layouts, dtype=float),
        days=np.asarray(days, dtype=int),
        localized_counts=count_matrix,
        localized_mean=np.asarray(means, dtype=float),
        localized_std=np.asarray(stds, dtype=float),
    )
    return load_npz_cache(path, rebuild_cache=False)


def panel_e_density_summary(
    paths: Figure2Paths, output_dir: Path, png: bool, cache_dir: Path, rebuild_cache: bool
) -> None:
    data = get_panel_e_data(paths, cache_dir, rebuild_cache)
    localized_layout_data = get_panel_e_localized_layout_data(paths, cache_dir, rebuild_cache)
    localized_layout_data_all5 = get_panel_e_localized_layout_data_all5(paths, cache_dir, rebuild_cache)
    x = data["layouts"]
    yield_values = data["yield_values"]
    yield_mean = np.nanmean(yield_values, axis=1)
    yield_std = np.nanstd(yield_values, axis=1)

    def plot_summary(values: np.ndarray, mean: np.ndarray, std: np.ndarray, ylabel: str, stem: str, ylim=None):
        fig, ax = plt.subplots(figsize=(2.2, 2.45))
        ax.fill_between(x, mean - std, mean + std, color="0.68", alpha=0.9, linewidth=0)
        ax.plot(x, mean, color="0.25", lw=1.0)
        for idx, layout in enumerate(x):
            vals = values[idx]
            valid = np.isfinite(vals)
            if np.any(valid):
                valid_vals = vals[valid]
                jitter = np.linspace(-1.8, 1.8, valid_vals.size) if valid_vals.size > 1 else np.zeros(valid_vals.size)
                ax.scatter(layout + jitter, valid_vals, color="0.35", s=8, alpha=0.75, linewidths=0, zorder=2)
        ax.set_xlabel("Electrode layouts", fontsize=9, labelpad=1)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
        ax.tick_params(axis="both", labelsize=8, direction="out", length=3, width=0.8, pad=1)
        ax.set_xlim(38, 133)
        ax.set_xticks([40, 60, 80, 100, 120])
        if ylim is not None:
            ax.set_ylim(*ylim)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.25")
        save_figure(fig, output_dir, stem, png=png)

    plot_summary(yield_values, yield_mean, yield_std, "Single unit yield", "fig2e_single_unit_yield", ylim=(20, 85))
    plot_localized_electrodes(localized_layout_data, output_dir, png, "fig2e_localized_electrodes")
    plot_localized_electrodes(localized_layout_data_all5, output_dir, png, "fig2e_localized_electrodes_all5_layouts")


def plot_localized_electrodes(data: dict[str, np.ndarray], output_dir: Path, png: bool, stem: str) -> None:
    x = data["layouts"]
    values = data["localized_counts"]
    mean = data["localized_mean"]
    std = data["localized_std"]

    fig, ax = plt.subplots(figsize=(2.2, 2.45))
    ax.fill_between(x, mean - std, mean + std, color="0.68", alpha=0.9, linewidth=0)
    ax.plot(x, mean, color="0.25", lw=1.0)

    for idx, layout in enumerate(x):
        vals = values[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            jitter = equal_spaced_offsets_by_value(vals, width=5.2)
            ax.scatter(
                layout + jitter,
                vals,
                color="0.35",
                alpha=0.65,
                s=2.0,
                linewidths=0,
                zorder=2,
            )

    ax.set_xlabel("Electrode layouts", fontsize=9, labelpad=1)
    ax.set_ylabel("Localized contacts", fontsize=9, labelpad=2)
    ax.tick_params(axis="both", labelsize=8, direction="out", length=3, width=0.8, pad=1)
    ax.set_xlim(38, 133)
    ax.set_ylim(0, 15)
    ax.set_xticks([40, 60, 80, 100, 120])
    ax.set_yticks(np.arange(0, 15, 2))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    save_figure(fig, output_dir, stem, png=png)


def equal_spaced_offsets_by_value(vals: np.ndarray, width: float = 5.2) -> np.ndarray:
    """Spread identical y-values evenly so integer-count points are visible."""
    offsets = np.zeros(vals.size, dtype=float)
    for value in np.unique(vals):
        idx = np.flatnonzero(vals == value)
        if idx.size > 1:
            offsets[idx] = np.linspace(-width / 2, width / 2, idx.size)
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        nargs="+",
        default=["a", "b", "c", "d"],
        choices=["a", "b", "c", "d"],
        help="Panel(s) to generate.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore existing lightweight panel caches and regenerate them from SpikeInterface data.",
    )
    parser.add_argument("--png", action="store_true", help="Also save PNG copies.")
    parser.add_argument(
        "--selected-ch",
        type=int,
        nargs="+",
        default=[86, 119],
        help="Channel index/indices for panel c. Defaults to two electrode-array sections: 86 and 119.",
    )
    parser.add_argument("--hanqing-folder", type=Path, default=Figure2Paths.hanqing_folder)
    parser.add_argument("--tracking-128-folder", type=Path, default=Figure2Paths.tracking_128_folder)
    parser.add_argument("--tracking-85-day1-folder", type=Path, default=Figure2Paths.tracking_85_day1_folder)
    parser.add_argument("--tracking-43-day1-folder", type=Path, default=Figure2Paths.tracking_43_day1_folder)
    parser.add_argument("--data-sorted-folder", type=Path, default=Figure2Paths.data_sorted_folder)
    return parser.parse_args()


def requested_panels(panel_args: Iterable[str]) -> list[str]:
    panels = list(panel_args)
    if "all" in panels:
        return ["a", "b", "c", "d", "e"]
    return panels


def main() -> None:
    args = parse_args()
    paths = Figure2Paths(
        hanqing_folder=args.hanqing_folder,
        tracking_128_folder=args.tracking_128_folder,
        tracking_85_day1_folder=args.tracking_85_day1_folder,
        tracking_43_day1_folder=args.tracking_43_day1_folder,
        data_sorted_folder=args.data_sorted_folder,
    )

    panel_functions = {
        "a": lambda: panel_a_raw_traces(paths, args.output_dir, args.png, args.cache_dir, args.rebuild_cache),
        "b": lambda: panel_b_unit_locations(paths, args.output_dir, args.png, args.cache_dir, args.rebuild_cache),
        "c": lambda: panel_c_waveform_distribution(
            paths, args.output_dir, args.png, args.cache_dir, args.rebuild_cache, selected_ch=args.selected_ch
        ),
        "d": lambda: panel_d_layout_connectivity(paths, args.output_dir, args.png, args.cache_dir, args.rebuild_cache),
        "e": lambda: panel_e_density_summary(paths, args.output_dir, args.png, args.cache_dir, args.rebuild_cache),
    }

    start = time.time()
    for panel in requested_panels(args.panel):
        panel_start = time.time()
        print(f"generating panel {panel}...")
        panel_functions[panel]()
        print(f"panel {panel} finished in {time.time() - panel_start:.1f}s")
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
