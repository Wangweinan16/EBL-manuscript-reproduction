from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "Arial"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from scipy import stats
from scipy.optimize import curve_fit


CONNECTIVITY_BY_CHANNEL = {
    13: [(136, 146)],
    29: [(104, 31), (31, 124), (31, 147), (104, 98), (98, 136), (104, 124), (106, 136)],
    39: [(98, 117), (104, 117), (104, 138)],
    45: [],
    84: [
        (106, 128), (136, 106), (106, 145), (128, 121), (146, 121), (136, 124),
        (142, 124), (146, 124), (136, 127), (146, 127), (136, 143), (136, 146),
        (143, 138), (146, 138), (142, 145), (146, 142), (146, 145),
    ],
    90: [(138, 96), (143, 96), (106, 128), (107, 146), (136, 128), (138, 128), (143, 128), (136, 138), (136, 143)],
    94: [(128, 121), (146, 121)],
    101: [(98, 86), (105, 98), (98, 136), (105, 104), (104, 136), (105, 106), (105, 121), (105, 124), (105, 136), (105, 149), (124, 136), (149, 124)],
    107: [(104, 98), (106, 98), (149, 98), (104, 117), (128, 104), (104, 136), (149, 104), (128, 117), (124, 136), (128, 136), (128, 149)],
    117: [
        (31, 98), (142, 31), (146, 31), (34, 136), (104, 98), (117, 98),
        (98, 138), (142, 98), (146, 105), (106, 136), (106, 138), (106, 142),
        (106, 145), (106, 149), (117, 142), (146, 117), (128, 136), (128, 138),
        (136, 138), (142, 136), (146, 136), (149, 136), (142, 138), (143, 138),
        (145, 138), (146, 138), (149, 138), (146, 142), (146, 145), (146, 149),
    ],
}

ACTIVATED_UNIT_COLOR = "#D62728"
INACTIVE_UNIT_COLOR = "#808080"
PROBE_UNIT_ALPHA = 0.52
CIRCLE_LABELS_TO_SHOW = {1, 5, 10, 15, 20, 25, 30}
PANEL_C_FONT = "Arial"
CONNECTIVITY_CIRCLE_FIGSIZE = (1.12, 1.12)
CONNECTIVITY_UNIT_DOT_SIZE = 3
CONNECTIVITY_STIM_DOT_SIZE = 3
PROBE_CIRCLE_LINEWIDTH = 0.25
CIRCLE_UNIT_LINEWIDTH = 0.05
CIRCLE_UNIT_EDGE_ALPHA = 0.35
CIRCLE_OUTLINE_LINEWIDTH = 0.5
CONNECTION_LINEWIDTH = 0.75
UNIT_LABEL_FONTSIZE = 6
MANUSCRIPT_CHANNEL_ORDER = [90, 84, 117, 13, 94, 107, 101, 29, 39, 45]
MANUSCRIPT_CHANNEL_MAP = {channel: idx for idx, channel in enumerate(MANUSCRIPT_CHANNEL_ORDER, start=1)}
ORIGINAL_COLOR_CHANNEL_ORDER = sorted(CONNECTIVITY_BY_CHANNEL.keys())
BASELINE_COLOR = "lightgray"




@dataclass
class StimData:
    currents: np.ndarray
    channels: np.ndarray
    timestamps: np.ndarray


class LightweightSorting:
    def __init__(self, spike_trains: dict[int, np.ndarray], sampling_frequency: float = 30000.0):
        self._spike_trains = {int(k): np.sort(np.asarray(v, dtype=np.int64)) for k, v in spike_trains.items()}
        self.unit_ids = np.array(sorted(self._spike_trains), dtype=int)
        self._sampling_frequency = float(sampling_frequency)

    def get_unit_ids(self) -> np.ndarray:
        return self.unit_ids

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_unit_spike_train(self, unit_id: int, start_frame=None, end_frame=None) -> np.ndarray:
        train = self._spike_trains[int(unit_id)]
        lo = 0 if start_frame is None else np.searchsorted(train, int(start_frame), side="left")
        hi = len(train) if end_frame is None else np.searchsorted(train, int(end_frame), side="left")
        return train[lo:hi]


class LightweightWaveforms:
    def __init__(self, folder: Path, sorting: LightweightSorting, unit_ids: np.ndarray, channel_locations: np.ndarray):
        self.folder = folder
        self.sorting = sorting
        self.unit_ids = np.asarray(unit_ids, dtype=int)
        self._channel_locations = channel_locations
        self.unit_locations = np.load(folder / "unit_locations" / "unit_locations.npy")
        self.templates = np.load(folder / "templates_average.npy", mmap_mode="r")
        self.templates_std = np.load(folder / "templates_std.npy", mmap_mode="r")
        self._unit_index = {int(unit_id): idx for idx, unit_id in enumerate(self.unit_ids)}

    def get_channel_locations(self) -> np.ndarray:
        return self._channel_locations

    def get_waveforms(self, unit_id: int) -> np.ndarray:
        path = self.folder / "waveforms" / f"waveforms_{int(unit_id)}.npy"
        return np.load(path, mmap_mode="r")

    def template_for(self, unit_id: int) -> np.ndarray:
        return np.asarray(self.templates[self._unit_index[int(unit_id)]])


def load_stim_data(data_folder: Path) -> StimData:
    stim_times_arr = sio.loadmat(data_folder / "stim_times_arr.mat")["stim_times_arr"]
    stim_key = sio.loadmat(data_folder / "stim_params.mat")["trials"]
    stim_times_dict = {
        stim_times_arr[i, 0][0][0]: list(stim_times_arr[i, 1].flatten())
        for i in range(stim_times_arr.shape[0])
    }
    currents = np.repeat(stim_key[:, 0].flatten(), 50)
    channels = np.repeat(stim_key[:, 1].flatten(), 50)
    timestamps = np.array([ts for timestamps in stim_times_dict.values() for ts in timestamps], dtype=np.int64)
    timestamps.sort()
    return StimData(currents=currents, channels=channels, timestamps=timestamps)


def collect_sorting_nodes(node: dict, out: list[dict]) -> None:
    if isinstance(node, dict) and "class" in node:
        out.append(node)
        parent = node.get("kwargs", {}).get("parent_sorting")
        if isinstance(parent, dict):
            collect_sorting_nodes(parent, out)


def load_channel_locations(waveform_folder: Path) -> np.ndarray:
    probegroup = json.loads((waveform_folder / "recording_info" / "probegroup.json").read_text())
    positions = probegroup["probes"][0]["contact_positions"]
    return np.asarray(positions, dtype=float)


def load_lightweight_waveforms(waveform_folder: Path) -> LightweightWaveforms:
    sorting_json = json.loads((waveform_folder / "curation_graph.json").read_text())
    nodes: list[dict] = []
    collect_sorting_nodes(sorting_json, nodes)
    npz_node = next(node for node in nodes if node["class"].endswith("NpzSortingExtractor"))
    firings_path = waveform_folder / npz_node["kwargs"]["file_path"]
    firings = np.load(firings_path, allow_pickle=False)
    sampling_frequency = float(firings["sampling_frequency"][0])
    raw_trains = {
        int(unit): firings["spike_indexes_seg0"][firings["spike_labels_seg0"] == int(unit)]
        for unit in firings["unit_ids"]
    }

    trains = raw_trains
    # Apply operations from raw extractor upward to the final curated selection.
    for node in reversed(nodes):
        cls = node["class"]
        kwargs = node.get("kwargs", {})
        if cls.endswith("UnitsSelectionSorting"):
            selected = [int(u) for u in kwargs["unit_ids"]]
            renamed = [int(u) for u in kwargs.get("renamed_unit_ids", selected)]
            trains = {renamed_i: trains[selected_i] for selected_i, renamed_i in zip(selected, renamed)}
        elif cls.endswith("MergeUnitsSorting"):
            for units_to_merge, new_unit_id in zip(kwargs["units_to_merge"], kwargs["new_unit_ids"]):
                merged = np.concatenate([trains[int(unit)] for unit in units_to_merge if int(unit) in trains])
                for unit in units_to_merge:
                    trains.pop(int(unit), None)
                trains[int(new_unit_id)] = np.sort(merged)

    final_unit_ids = np.array(sorting_json["kwargs"]["unit_ids"], dtype=int)
    sorting = LightweightSorting({int(unit): trains[int(unit)] for unit in final_unit_ids}, sampling_frequency)
    return LightweightWaveforms(waveform_folder, sorting, final_unit_ids, load_channel_locations(waveform_folder))


def compute_unit_locations(we: LightweightWaveforms) -> np.ndarray:
    return we.unit_locations


def activation_pairs(
    we: LightweightWaveforms,
    stim: StimData,
    p_threshold: float = 0.001,
    require_mean_increase: bool = True,
) -> pd.DataFrame:
    pairs: list[tuple[int, int, float]] = []
    for unit_id in we.unit_ids:
        for channel in np.unique(stim.channels):
            for current in np.unique(stim.currents):
                timestamps = stim.timestamps[(stim.currents == current) & (stim.channels == channel)]
                if len(timestamps) < 50:
                    continue
                stim_rates = []
                baseline_rates = []
                for start_idx in range(0, min(500, len(timestamps)), 50):
                    start = int(timestamps[start_idx])
                    end = int(timestamps[min(start_idx + 49, len(timestamps) - 1)])
                    duration = (end - start) / 30000.0
                    if duration <= 0:
                        continue
                    stim_count = len(we.sorting.get_unit_spike_train(unit_id, start, end))
                    base_count = len(we.sorting.get_unit_spike_train(unit_id, start - 30000, start))
                    stim_rates.append(stim_count / duration)
                    baseline_rates.append(base_count / duration)
                if len(stim_rates) < 2:
                    continue
                _t, p_val = stats.ttest_ind(stim_rates, baseline_rates, equal_var=False)
                mean_ok = np.mean(stim_rates) >= np.mean(baseline_rates)
                if np.isfinite(p_val) and p_val <= p_threshold and (mean_ok or not require_mean_increase):
                    pairs.append((int(unit_id), int(channel), float(current)))
    return pd.DataFrame(pairs, columns=["unit_id", "channel", "current"])


def manuscript_channel_label(raw_channel: int) -> str:
    return f"Ch {MANUSCRIPT_CHANNEL_MAP[int(raw_channel)]}"


def channel_mapping_table(we: LightweightWaveforms, stim: StimData) -> pd.DataFrame:
    channel_locations = we.get_channel_locations()
    rows = []
    for raw_channel in MANUSCRIPT_CHANNEL_ORDER:
        if raw_channel not in set(int(ch) for ch in np.unique(stim.channels)):
            continue
        loc = channel_locations[raw_channel - 1]
        rows.append(
            {
                "display_channel": MANUSCRIPT_CHANNEL_MAP[raw_channel],
                "display_label": manuscript_channel_label(raw_channel),
                "raw_channel": raw_channel,
                "x": float(loc[0]),
                "y": float(loc[1]),
            }
        )
    return pd.DataFrame(rows)


def write_manuscript_channel_mapping(we: LightweightWaveforms, stim: StimData, output_dir: Path) -> pd.DataFrame:
    mapping = channel_mapping_table(we, stim)
    mapping.to_csv(output_dir / "manuscript_channel_mapping_spatial_descending_y.csv", index=False)
    (output_dir / "manuscript_channel_mapping_spatial_descending_y.json").write_text(
        json.dumps(mapping.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return mapping


def original_channel_colors() -> dict[int, tuple[float, float, float]]:
    palette = sns.color_palette("deep", len(ORIGINAL_COLOR_CHANNEL_ORDER))
    return dict(zip(ORIGINAL_COLOR_CHANNEL_ORDER, palette))


def manuscript_channel_colors() -> dict[int, tuple[float, float, float]]:
    return original_channel_colors()


def save_manuscript_channel_legend(
    colors: dict[int, tuple[float, float, float]],
    output_dir: Path,
    filename: str,
    *,
    ncol: int = 2,
    include_baseline: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    if figsize is None:
        figsize = (1.1, 0.75) if ncol > 1 else (0.62, 1.35)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    marker_size = 4 if include_baseline and ncol == 1 else 5
    marker_edge = "none"
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=marker_size, markerfacecolor=colors[channel], markeredgecolor=marker_edge)
        for channel in MANUSCRIPT_CHANNEL_ORDER
    ]
    labels = [manuscript_channel_label(channel) for channel in MANUSCRIPT_CHANNEL_ORDER]
    if include_baseline:
        handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=marker_size, markerfacecolor=BASELINE_COLOR, markeredgecolor=marker_edge)
        )
        labels.append("Baseline")
    legend = ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=not include_baseline,
        fancybox=False,
        edgecolor="0.35",
        fontsize=6,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.7 if ncol > 1 else 0.3,
        borderpad=0.35,
        labelspacing=0.25 if ncol > 1 else 0.18,
    )
    if not include_baseline:
        legend.get_frame().set_linewidth(0.8)
    fig.savefig(output_dir / filename, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_panel_a_probe_mapped(we: LightweightWaveforms, stim: StimData, output_dir: Path) -> None:
    unit_locations = compute_unit_locations(we)
    channel_locations = we.get_channel_locations()
    colors = manuscript_channel_colors()

    fig, ax = plt.subplots(figsize=(6, 9))
    for location in unit_locations:
        ax.scatter(location[0], location[1], color="gray", zorder=2, s=40)

    for channel in MANUSCRIPT_CHANNEL_ORDER:
        location = channel_locations[channel - 1]
        if 1 <= location[1] <= 2000:
            ax.scatter(*location, color=colors[channel], zorder=3, edgecolor="black", s=100, linewidth=0.5, alpha=0.7)
            ax.annotate(
                "",
                xy=location,
                xytext=(location[0], location[1] + 100),
                arrowprops=dict(edgecolor=colors[channel], facecolor=colors[channel], shrink=0.05, width=1, headwidth=8, headlength=10, linestyle="-", alpha=0.7),
                zorder=4,
            )

    for idx, loc in enumerate(channel_locations):
        if idx + 1 not in MANUSCRIPT_CHANNEL_ORDER:
            ax.scatter(*loc, color="none", s=50, edgecolors="black", linewidth=0.5, zorder=1)

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - 100, x_max + 100)
    ax.set_ylim([500, 2000])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "subplot_a1_mapped.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    save_manuscript_channel_legend(colors, output_dir, "subplot_a1_mapped_legend.svg")


def spatial_stim_channel_order(we: LightweightWaveforms, stim: StimData, descending_y: bool = True) -> list[int]:
    channel_locations = we.get_channel_locations()
    channels = [int(channel) for channel in np.unique(stim.channels)]
    if set(MANUSCRIPT_CHANNEL_ORDER) == set(channels) and descending_y:
        return MANUSCRIPT_CHANNEL_ORDER.copy()

    def sort_key(channel: int):
        loc = channel_locations[channel - 1]
        y_key = -float(loc[1]) if descending_y else float(loc[1])
        return (y_key, float(loc[0]), channel)

    return sorted(channels, key=sort_key)


def save_panel_b_raster_spatial(we: LightweightWaveforms, stim: StimData, output_dir: Path) -> None:
    unit_locations = compute_unit_locations(we)
    sorted_unit_ids = [unit for unit, _loc in sorted(zip(we.unit_ids, unit_locations), key=lambda item: item[1][1], reverse=True)]
    channels = spatial_stim_channel_order(we, stim, descending_y=True)
    colors = manuscript_channel_colors()
    write_manuscript_channel_mapping(we, stim, output_dir)

    fig, axes = plt.subplots(
        1,
        len(channels),
        figsize=(5.35, 2.85),
        sharey=True,
        gridspec_kw={"wspace": 0.035},
    )
    if len(channels) == 1:
        axes = [axes]

    unit_spacing = 2.5
    block_low = -0.5
    block_high = 1.5
    y_min = block_low
    y_max = (len(sorted_unit_ids) - 1) * unit_spacing + block_high

    for ax, channel in zip(axes, channels):
        timestamps = stim.timestamps[(stim.currents == 5) & (stim.channels == channel)]
        if len(timestamps) < 50:
            continue
        stim_line_start = stim_line_end = None
        for unit_idx, unit_id in enumerate(sorted_unit_ids):
            line_offset = unit_idx * unit_spacing
            stim_rates = []
            baseline_rates = []
            for start_idx in range(0, min(500, len(timestamps)), 50):
                start = int(timestamps[start_idx])
                end = int(timestamps[min(start_idx + 49, len(timestamps) - 1)])
                trial_start = start - 30000
                trial_end = end + 30000
                stim_line_start = (start - trial_start) / 30000
                stim_line_end = (end - trial_start) / 30000
                trial_length = (trial_end - trial_start) / 30000
                ax.hlines(
                    y=line_offset + block_low,
                    xmin=0,
                    xmax=trial_length,
                    colors="0.82",
                    linestyles="dashed",
                    linewidth=0.32,
                    zorder=1,
                )
                ax.hlines(
                    y=line_offset + block_high,
                    xmin=0,
                    xmax=trial_length,
                    colors="0.82",
                    linestyles="dashed",
                    linewidth=0.32,
                    zorder=1,
                )
                spikes = we.sorting.get_unit_spike_train(unit_id, trial_start, trial_end)
                spike_times_sec = (spikes - trial_start) / 30000
                y_offsets = np.full(spike_times_sec.shape, line_offset + start_idx / 500, dtype=float)
                ax.scatter(
                    spike_times_sec,
                    y_offsets,
                    color="black",
                    s=0.42,
                    linewidths=0,
                    edgecolors="none",
                    zorder=3,
                )
                duration = (end - start) / 30000
                stim_rates.append(len(we.sorting.get_unit_spike_train(unit_id, start, end)) / duration)
                baseline_rates.append(len(we.sorting.get_unit_spike_train(unit_id, start - 30000, start)) / duration)
            _t, p_val = stats.ttest_ind(stim_rates, baseline_rates, equal_var=False)
            if np.isfinite(p_val) and p_val <= 0.005 and np.mean(stim_rates) >= np.mean(baseline_rates):
                ax.axhspan(line_offset + block_low, line_offset + block_high, color=colors[channel], alpha=0.35, zorder=0)

        if stim_line_start is not None:
            ax.axvline(stim_line_start, color=colors[channel], linestyle="--", linewidth=0.8, zorder=4)
            ax.axvline(stim_line_end, color=colors[channel], linestyle="--", linewidth=0.8, zorder=4)
        ax.set_title(manuscript_channel_label(channel), fontsize=7, pad=3)
        ax.set_xlim(0, 3)
        ax.set_ylim(y_max, y_min)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0, pad=2, labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        for side in ("left", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.45)
        ax.margins(x=0, y=0)

    row_labels = [1, 10, 20, 30, len(sorted_unit_ids)]
    row_positions = [(row - 1) * unit_spacing + 0.5 for row in row_labels]
    axes[0].set_yticks(row_positions)
    axes[0].set_yticklabels([str(row) for row in row_labels], fontsize=6)
    axes[0].set_ylabel("Neurons", fontsize=7, labelpad=8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.04, wspace=0.035)
    fig.savefig(output_dir / "5uA_lineoffset_4b_spatial_channels.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def map_depth_to_angle(depth, min_depth=500, max_depth=2000):
    return 2 * np.pi * (depth - min_depth) / (max_depth - min_depth)


def unit_depth_to_circle_positions(unit_locations, radius=9):
    depths = np.array([loc[1] for loc in unit_locations])
    angles = map_depth_to_angle(depths)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles))), angles


def active_units_for_channel(unit_channel: pd.DataFrame, channel: int) -> set[int]:
    return set(unit_channel.loc[unit_channel["channel"] == channel, "unit_id"].astype(int))


def panel_c_mapped_channel_colors() -> dict[int, tuple[float, float, float]]:
    return manuscript_channel_colors()


def save_panel_c_probe_columns_mapped(we: LightweightWaveforms, unit_channel: pd.DataFrame, output_dir: Path) -> None:
    panel_dir = output_dir / "panel_c" / "probe_columns_mapped"
    panel_no_numbers_dir = output_dir / "panel_c" / "probe_columns_mapped_no_numbers"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_no_numbers_dir.mkdir(parents=True, exist_ok=True)
    unit_locations = compute_unit_locations(we)
    channel_locations = we.get_channel_locations()
    unit_order = sorted(zip(unit_locations, we.unit_ids), key=lambda item: item[0][1], reverse=True)

    def draw_probe_column(channel: int, show_unit_numbers: bool):
        active_units = active_units_for_channel(unit_channel, channel)
        fig, ax = plt.subplots(figsize=(1.55, 8))

        for site_idx, site_location in enumerate(channel_locations, start=1):
            if 500 <= site_location[1] <= 2000 and site_idx != channel:
                ax.scatter(
                    site_location[0],
                    site_location[1],
                    color="none",
                    edgecolor="black",
                    linewidth=PROBE_CIRCLE_LINEWIDTH,
                    s=28,
                    zorder=1,
                )

        for row_idx, (location, unit_id) in enumerate(unit_order, start=1):
            is_active = int(unit_id) in active_units
            ax.scatter(
                location[0],
                location[1],
                color=ACTIVATED_UNIT_COLOR if is_active else INACTIVE_UNIT_COLOR,
                edgecolor="black",
                linewidth=PROBE_CIRCLE_LINEWIDTH,
                s=52 if is_active else 38,
                alpha=PROBE_UNIT_ALPHA,
                zorder=3 if is_active else 2,
            )
            if show_unit_numbers:
                ax.text(
                    location[0] - 36,
                    location[1],
                    str(row_idx),
                    ha="right",
                    va="center",
                    fontsize=UNIT_LABEL_FONTSIZE,
                    fontname=PANEL_C_FONT,
                    color="black",
                )

        ax.set_xlim(unit_locations[:, 0].min() - 90, unit_locations[:, 0].max() + 45)
        ax.set_ylim(500, 2000)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        out_dir = panel_dir if show_unit_numbers else panel_no_numbers_dir
        suffix = "" if show_unit_numbers else "_no_numbers"
        display_channel = MANUSCRIPT_CHANNEL_MAP[channel]
        fig.savefig(out_dir / f"panel_d_probe_ch{display_channel}_rawch{channel}{suffix}.svg", format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)

    for channel in MANUSCRIPT_CHANNEL_ORDER:
        draw_probe_column(channel, show_unit_numbers=True)
        draw_probe_column(channel, show_unit_numbers=False)


def save_panel_c_connectivity_mapped(we: LightweightWaveforms, unit_channel: pd.DataFrame, output_dir: Path) -> None:
    panel_dir = output_dir / "panel_c" / "connectivity_circles_mapped"
    panel_dir.mkdir(parents=True, exist_ok=True)
    unit_locations = compute_unit_locations(we)
    channel_locations = we.get_channel_locations()
    colors = panel_c_mapped_channel_colors()
    unit_positions, angles = unit_depth_to_circle_positions(unit_locations)
    label_one_angle = max(zip(angles, we.unit_ids), key=lambda item: item[0])[0]
    angle_offset = (np.pi / 2) - label_one_angle
    rotated_angles = angles + angle_offset
    unit_positions = np.column_stack((9 * np.cos(rotated_angles), 9 * np.sin(rotated_angles)))
    unit_id_to_index = {int(unit_id): idx for idx, unit_id in enumerate(we.sorting.get_unit_ids())}

    def draw_connections(ax, connections, color):
        for start_unit_id, end_unit_id in connections:
            start_idx = unit_id_to_index.get(int(start_unit_id))
            end_idx = unit_id_to_index.get(int(end_unit_id))
            if start_idx is None or end_idx is None:
                continue
            start_pos = unit_positions[start_idx]
            end_pos = unit_positions[end_idx]
            control = ((start_pos + end_pos) / 2) * 0.5
            path = MplPath(np.array([start_pos, control, end_pos]), [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
            ax.add_patch(patches.PathPatch(path, facecolor="none", edgecolor=color, lw=CONNECTION_LINEWIDTH, alpha=0.7))
            ax.add_patch(FancyArrowPatch(posA=end_pos - 0.001 * (end_pos - control), posB=end_pos, arrowstyle="->", color=color, linewidth=CONNECTION_LINEWIDTH, mutation_scale=6, alpha=1))

    for channel in MANUSCRIPT_CHANNEL_ORDER:
        connections = CONNECTIVITY_BY_CHANNEL[channel]
        active_units = active_units_for_channel(unit_channel, channel)
        fig, ax = plt.subplots(figsize=CONNECTIVITY_CIRCLE_FIGSIZE)
        ax.add_artist(plt.Circle((0, 0), 10, color="black", fill=False, linewidth=CIRCLE_OUTLINE_LINEWIDTH))
        sorted_units = sorted(zip(unit_positions, rotated_angles, angles, we.unit_ids), key=lambda item: item[2], reverse=True)
        for idx, (pos, rotated_angle, _original_angle, unit_id) in enumerate(sorted_units, start=1):
            active = int(unit_id) in active_units
            ax.scatter(
                pos[0],
                pos[1],
                color=ACTIVATED_UNIT_COLOR if active else INACTIVE_UNIT_COLOR,
                edgecolor="none",
                linewidth=0,
                zorder=2 if active else 1,
                s=CONNECTIVITY_UNIT_DOT_SIZE,
            )
            if idx in CIRCLE_LABELS_TO_SHOW:
                ax.text(
                    11 * np.cos(rotated_angle),
                    11 * np.sin(rotated_angle),
                    str(idx),
                    color="black",
                    fontsize=UNIT_LABEL_FONTSIZE,
                    fontname=PANEL_C_FONT,
                    ha="center",
                    va="center",
                    zorder=4,
                )
        filtered = [
            (start, end)
            for start, end in connections
            if int(start) in active_units and int(end) in active_units
        ]
        draw_connections(ax, filtered, colors[channel])
        loc = channel_locations[channel - 1]
        if 500 <= loc[1] <= 2000:
            angle = map_depth_to_angle(loc[1]) + angle_offset
            ax.scatter(
                9 * np.cos(angle),
                9 * np.sin(angle),
                color=colors[channel],
                zorder=3,
                edgecolor=(0, 0, 0, CIRCLE_UNIT_EDGE_ALPHA),
                s=CONNECTIVITY_STIM_DOT_SIZE,
                linewidth=CIRCLE_UNIT_LINEWIDTH,
            )
        ax.set_aspect("equal")
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.axis("off")
        display_channel = MANUSCRIPT_CHANNEL_MAP[channel]
        fig.savefig(panel_dir / f"panel_f_circle_ch{display_channel}_rawch{channel}.svg", format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)

    save_manuscript_channel_legend(
        colors,
        output_dir,
        "panel_f_mapped_legend.svg",
        ncol=1,
        include_baseline=True,
        figsize=(0.7, 1.45),
    )


def collect_umap_trial_vectors(we: LightweightWaveforms, stim: StimData) -> tuple[np.ndarray, dict[int, list[list[float]]], list[list[float]]]:
    channels = np.unique(stim.channels)
    trial_vectors_by_channel = {}
    combined_baseline_data = []
    for channel in channels:
        timestamps = stim.timestamps[stim.channels == channel]
        trial_vectors = []
        for start_idx in range(0, min(2500, len(timestamps)), 50):
            start = int(timestamps[start_idx])
            end = int(timestamps[min(start_idx + 49, len(timestamps) - 1)])
            duration = (end - start) / 30000
            stim_rates = []
            baseline_rates = []
            for unit_id in we.unit_ids:
                stim_rates.append(len(we.sorting.get_unit_spike_train(unit_id, start, end)) / duration)
                baseline_rates.append(len(we.sorting.get_unit_spike_train(unit_id, start - 30000, start)))
            trial_vectors.append(stim_rates)
            combined_baseline_data.append(baseline_rates)
        trial_vectors_by_channel[int(channel)] = trial_vectors
    return channels, trial_vectors_by_channel, combined_baseline_data


def save_panel_d_umap_manuscript_svgs(we: LightweightWaveforms, stim: StimData, output_dir: Path, random_states: Iterable[int]) -> None:
    import umap

    manuscript_dir = output_dir / "umap_manuscript_svgs"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    channels, trial_vectors_by_channel, combined_baseline_data = collect_umap_trial_vectors(we, stim)
    colors = dict(zip(channels, sns.color_palette("deep", len(channels))))

    def ticks_at_multiples_of_five(limits):
        lo = int(np.floor(min(limits) / 5) * 5)
        hi = int(np.ceil(max(limits) / 5) * 5)
        return np.arange(lo, hi + 1, 5)

    def rounded_limits_and_ticks(limits):
        ticks = ticks_at_multiples_of_five(limits)
        return (float(ticks[0]), float(ticks[-1])), ticks

    def draw_back_box_edges(ax):
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xs = [xlim[0], xlim[1]]
        ys = [ylim[0], ylim[1]]
        zs = [zlim[0], zlim[1]]
        corners = {
            (ix, iy, iz): np.array([xs[ix], ys[iy], zs[iz]], dtype=float)
            for ix in [0, 1]
            for iy in [0, 1]
            for iz in [0, 1]
        }

        elev = np.deg2rad(ax.elev)
        azim = np.deg2rad(ax.azim)
        view = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        center = np.array([np.mean(xlim), np.mean(ylim), np.mean(zlim)])
        front_corner = max(corners, key=lambda key: float(np.dot(corners[key] - center, view)))

        edges = []
        for iy in [0, 1]:
            for iz in [0, 1]:
                edges.append(("x", (0, iy, iz), (1, iy, iz)))
        for ix in [0, 1]:
            for iz in [0, 1]:
                edges.append(("y", (ix, 0, iz), (ix, 1, iz)))
        for ix in [0, 1]:
            for iy in [0, 1]:
                edges.append(("z", (ix, iy, 0), (ix, iy, 1)))

        for _direction, start_key, end_key in edges:
            if start_key == front_corner or end_key == front_corner:
                continue
            start = corners[start_key]
            end = corners[end_key]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                linestyle="--",
                linewidth=0.35,
                color=(0.6, 0.6, 0.6, 0.65),
                zorder=0,
            )

    for random_state in random_states:
        all_reduced_points = []
        for channel in channels:
            reduced = umap.UMAP(n_components=3, random_state=int(random_state)).fit_transform(trial_vectors_by_channel[int(channel)])
            all_reduced_points.append((int(channel), reduced))
        baseline_reduced = umap.UMAP(n_components=3, random_state=int(random_state)).fit_transform(combined_baseline_data)

        fig = plt.figure(figsize=(2.7, 2.56))
        ax = fig.add_subplot(111, projection="3d")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis._axinfo["grid"]["linestyle"] = "--"
            axis._axinfo["grid"]["linewidth"] = 0
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            axis._axinfo["axisline"]["linewidth"] = 0.4
            axis._axinfo["axisline"]["color"] = (0.4, 0.4, 0.4, 0.8)
        ax.grid(False)
        for channel, points in all_reduced_points:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors[channel], label=f"Ch{channel}", alpha=0.7, s=1.5, edgecolors="none", linewidths=0)
        ax.scatter(baseline_reduced[:, 0], baseline_reduced[:, 1], baseline_reduced[:, 2], color="lightgray", alpha=0.5, label="Baseline", s=1.5, edgecolors="none", linewidths=0)
        xlim, xticks = rounded_limits_and_ticks(ax.get_xlim3d())
        ylim, yticks = rounded_limits_and_ticks(ax.get_ylim3d())
        zlim, zticks = rounded_limits_and_ticks(ax.get_zlim3d())
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks)
        draw_back_box_edges(ax)
        ax.set_xlabel("UMAP 1", fontsize=6, labelpad=-2)
        ax.set_ylabel("UMAP 2", fontsize=6, labelpad=-2)
        ax.set_zlabel("UMAP 3", fontsize=6, labelpad=-2)
        ax.tick_params(axis="both", labelsize=5, pad=-2)
        ax.zaxis.set_tick_params(labelsize=5, pad=-2)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=5, frameon=False, markerscale=1.5)
        fig.savefig(manuscript_dir / f"umap_allcurrents5_seed_{int(random_state)}_manuscript.svg", format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot the representative ICMS session from sorter output and curation graph.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sorting", choices=("figure", "native"), default="figure", help="Use the saved figure renderer merge rules or the native SpikeInterface 0.99.1 curated trains.")
    parser.add_argument("--umap", action="store_true", help="Also run the manuscript seed-97 UMAP (requires umap-learn).")
    args = parser.parse_args()
    data = Path(__file__).resolve().parent / "data"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stim = load_stim_data(data)
    we = load_lightweight_waveforms(data)
    if args.sorting == "native":
        with np.load(data / "native_curated_spike_frames.npz") as native:
            trains = {int(unit): native[str(int(unit))] for unit in we.unit_ids}
            we.sorting = LightweightSorting(trains, float(native["sampling_frequency"]))
    pairs = activation_pairs(we, stim)
    pairs.to_csv(output / "activation_pairs.csv", index=False)
    save_panel_a_probe_mapped(we, stim, output)
    save_panel_b_raster_spatial(we, stim, output)
    save_panel_c_probe_columns_mapped(we, pairs, output)
    save_panel_c_connectivity_mapped(we, pairs, output)
    if args.umap:
        os.environ.setdefault("NUMBA_CACHE_DIR", str((output / ".numba_cache").resolve()))
        save_panel_d_umap_manuscript_svgs(we, stim, output, [97])
