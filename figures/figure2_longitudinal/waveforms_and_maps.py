from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle


REPO_FIGURE3 = Path(__file__).resolve().parent
DEFAULT_WAVEFORM_ROOT = REPO_FIGURE3 / 'data'
DEFAULT_OUTPUT_DIR = REPO_FIGURE3 / "regenerated_figure3"
FIG3D_E_SIZE = (3.6, 1.68)

CONNECTIONS_DAY1 = [
    (4, 5), (5, 7), (14, 12), (15, 14), (18, 20), (27, 31), (29, 30),
    (30, 55), (54, 40), (54, 42), (54, 59), (59, 74), (65, 88), (82, 74),
    (84, 80), (88, 89), (89, 86), (89, 90), (91, 90), (93, 91), (93, 95),
]
CONNECTIONS_DAY5 = [
    (4, 5), (15, 14), (18, 17), (18, 20), (31, 29), (42, 38), (42, 40),
    (84, 71), (88, 89), (91, 90), (92, 84), (92, 95),
]
CONNECTIONS_DAY10 = [
    (7, 15), (12, 13), (14, 15), (15, 18), (16, 29), (30, 29), (40, 27),
    (42, 31), (71, 53), (88, 89), (91, 90), (93, 82),
]
CONNECTIONS_BY_PANEL = [CONNECTIONS_DAY1, CONNECTIONS_DAY5, CONNECTIONS_DAY10]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_day(folder: Path) -> dict:
    metrics = pd.read_csv(folder / "quality_metrics" / "metrics.csv", index_col=0)
    metrics.index = metrics.index.astype(int)
    locations = np.load(folder / "unit_locations" / "unit_locations.npy")
    templates = np.load(folder / "templates_average.npy")
    rec_attrs = read_json(folder / "recording_info" / "recording_attributes.json")
    channel_locations = np.asarray(rec_attrs["properties"]["location"], dtype=float)
    unit_ids = metrics.index.to_numpy(dtype=int)
    if len(unit_ids) != len(locations) or len(unit_ids) != len(templates):
        raise ValueError(f"Unit count mismatch in {folder}")
    return {
        "folder": folder,
        "metrics": metrics,
        "unit_ids": unit_ids,
        "unit_locations": locations,
        "templates": templates,
        "channel_locations": channel_locations,
    }


def load_days(waveform_root: Path) -> list[dict]:
    return [load_day(waveform_root / f"waveforms_sorting_{day}") for day in range(1, 11)]


def unit_color_map(days: list[dict]) -> dict[int, tuple[float, float, float, float]]:
    unique_unit_ids = sorted(set().union(*(set(day["unit_ids"]) for day in days)))
    palette = sns.color_palette("deep", len(unique_unit_ids))
    colors = mcolors.to_rgba_array(palette)
    return {unit_id: colors[i] for i, unit_id in enumerate(unique_unit_ids)}


def template_amplitude(template: np.ndarray) -> float:
    p2p_by_channel = np.ptp(template, axis=0)
    best_channel = int(np.argmax(p2p_by_channel))
    return float(np.max(np.abs(template[:, best_channel])))


def draw_tracking_panel(
    days: list[dict],
    output_path: Path,
    *,
    with_connections: bool,
    figsize: tuple[float, float],
    wspace: float = 1.0,
    min_firing_rate: float = 0.1,
    min_snr: float = 3.5,
) -> None:
    with matplotlib.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        selected_indices = [0, 4, 9]
        selected_days = [days[index] for index in selected_indices]
        color_dict = unit_color_map(days)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plt.subplots_adjust(wspace=wspace, hspace=0.5)
        all_channel_locations = selected_days[0]["channel_locations"]
        xlim = (-50, 50)
        ylim = (1000, 2000)
        titles = ["day1", "day5", "day10"]
        unit_locations_across_subplots = []

        if with_connections:
            counts = Counter(conn for conns in CONNECTIONS_BY_PANEL for conn in conns)
        else:
            counts = Counter()

        for ax_idx, (ax, day) in enumerate(zip(axes, selected_days)):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(titles[ax_idx], fontsize=6, pad=1)
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            ax.set_xticks([-40, 0, 40])
            ax.tick_params(axis="both", labelsize=5, width=0.6, length=2, pad=1)
            if ax_idx == 0:
                ax.get_yaxis().set_visible(True)
            else:
                ax.get_yaxis().set_visible(False)

            for loc in all_channel_locations:
                ax.scatter(*loc, color="black", s=4, zorder=2, edgecolors="black", linewidth=0.25)

            unit_id_to_index = {unit_id: idx for idx, unit_id in enumerate(day["unit_ids"])}
            if with_connections:
                for start_unit, end_unit in CONNECTIONS_BY_PANEL[ax_idx]:
                    if start_unit not in unit_id_to_index or end_unit not in unit_id_to_index:
                        continue
                    start_location = day["unit_locations"][unit_id_to_index[start_unit]]
                    end_location = day["unit_locations"][unit_id_to_index[end_unit]]
                    if start_location[1] <= 1000 or end_location[1] <= 1000:
                        continue
                    ax.add_patch(
                        FancyArrowPatch(
                            posA=start_location[:2],
                            posB=end_location[:2],
                            arrowstyle="->",
                            color="black",
                            linewidth=0.8,
                            connectionstyle="arc3,rad=0.2",
                            mutation_scale=7,
                            alpha=1,
                            clip_on=False,
                        )
                    )

            for idx, unit_id in enumerate(day["unit_ids"]):
                metrics = day["metrics"].loc[unit_id]
                location = day["unit_locations"][idx]
                if metrics["firing_rate"] < min_firing_rate or metrics["snr"] < min_snr or location[1] < 1000:
                    continue
                amp = template_amplitude(day["templates"][idx])
                color = color_dict[unit_id]
                ax.scatter(
                    location[0],
                    location[1],
                    color=color,
                    s=amp * 0.75,
                    zorder=3,
                    alpha=0.3,
                    edgecolors="black",
                    linewidth=0.5,
                    clip_on=False,
                )
                unit_locations_across_subplots.append((unit_id, ax_idx, location, color))

            contour_rect = plt.Rectangle(
                (-40, 1000),
                80,
                1000,
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
                clip_on=False,
                zorder=10,
            )
            ax.add_patch(contour_rect)

        if not with_connections:
            unit_locations_across_subplots.sort(key=lambda item: (item[0], item[1]))
            for current, next_item in zip(unit_locations_across_subplots, unit_locations_across_subplots[1:]):
                unit_id, prev_ax_idx, prev_location, color = current
                next_unit_id, ax_idx, location, _ = next_item
                if unit_id != next_unit_id or prev_ax_idx == ax_idx:
                    continue
                start = axes[prev_ax_idx].transData.transform(prev_location[:2])
                end = axes[ax_idx].transData.transform(location[:2])
                start = fig.transFigure.inverted().transform(start)
                end = fig.transFigure.inverted().transform(end)
                fig.lines.append(
                    mlines.Line2D(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color=color,
                        linewidth=0.8,
                        linestyle="--",
                        transform=fig.transFigure,
                        zorder=5,
                    )
                )

        for amp in [50, 100, 150]:
            axes[-1].scatter([], [], c="k", alpha=0.3, s=amp * 0.75, label=f"{amp}")
        axes[-1].legend(scatterpoints=1, frameon=True, labelspacing=1, loc="lower right", fontsize=5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg", transparent=True)
        plt.close(fig)


def draw_unit_waveforms(
    days: list[dict],
    output_dir: Path,
    unit_ids: list[int],
    *,
    waveform_linewidth: float = 0.3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    day_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    tracking_colors = unit_color_map(days)
    # Match the older Unit*_10days_3c rendered spacing, with slightly expanded
    # channel-group spacing for Illustrator placement stability.
    waveform_map_x_scale = 2.20 * 1.10 * 1.50
    waveform_map_y_scale = 5.95 * 1.10 * 1.50 * 1.20
    waveform_time_scale = 21.4
    waveform_amplitude_scale = 0.62 * 1.20
    day_waveform_y_offset = 12.0 * 1.50 * 1.50
    waveform_layout_channels = sorted(
        {17, 20, 22, 27, 29, 39, 41, 43, 45, 80, 82, 84, 86},
        key=lambda channel: (-days[0]["channel_locations"][channel, 1], days[0]["channel_locations"][channel, 0]),
    )
    plotted_waveform_channels = waveform_layout_channels[1:-1]
    channel_locations = days[0]["channel_locations"]
    layout_center_x = float(np.median([channel_locations[channel, 0] for channel in plotted_waveform_channels]))
    layout_center_y = float(np.median([channel_locations[channel, 1] for channel in plotted_waveform_channels]))
    plotted_locations = channel_locations[plotted_waveform_channels]
    base_x_min = min(
        layout_center_x + (location[0] - layout_center_x) * waveform_map_x_scale
        for location in plotted_locations
    ) - waveform_time_scale * 1.6
    base_x_max = max(
        layout_center_x + (location[0] - layout_center_x) * waveform_map_x_scale
        for location in plotted_locations
    ) + waveform_time_scale * 1.6
    base_y_min = min(
        layout_center_y + (location[1] - layout_center_y) * waveform_map_y_scale
        for location in plotted_locations
    ) - 180
    base_y_max = max(
        layout_center_y + (location[1] - layout_center_y) * waveform_map_y_scale
        for location in plotted_locations
    ) + 180
    waveform_xlim = (base_x_min, base_x_max)
    waveform_ylim = (base_y_min, base_y_max)
    waveform_figsize = (1.65, 2.65)
    with matplotlib.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        legend_fig, legend_ax = plt.subplots(figsize=waveform_figsize)
        legend_ax.set_xlim(*waveform_xlim)
        legend_ax.set_ylim(*waveform_ylim)
        legend_ax.axis("off")
        legend_handles = [
            mlines.Line2D([], [], color=day_colors[day_idx], linewidth=0.8, label=f"Day {day_idx + 1}")
            for day_idx in range(10)
        ]
        legend_ax.legend(
            handles=legend_handles,
            title="Recording Day",
            loc="upper left",
            frameon=False,
            fontsize=5,
            title_fontsize=5,
            labelspacing=0.65,
            handlelength=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
        )
        # Scale bar matched to the original tall Unit*_10days_3c panels.
        x0 = waveform_xlim[0] + 0.63 * (waveform_xlim[1] - waveform_xlim[0])
        y0 = waveform_ylim[0] + 0.16 * (waveform_ylim[1] - waveform_ylim[0])
        legend_ax.plot([x0, x0 + waveform_time_scale], [y0, y0], color="black", linewidth=1)
        legend_ax.plot([x0, x0], [y0, y0 + 100 * waveform_amplitude_scale], color="black", linewidth=1)
        legend_ax.text(x0 + waveform_time_scale / 2, y0 - 0.025 * (waveform_ylim[1] - waveform_ylim[0]), "1 ms", ha="center", va="top", fontsize=4)
        legend_ax.text(
            x0 - 0.035 * (waveform_xlim[1] - waveform_xlim[0]),
            y0 + 50 * waveform_amplitude_scale,
            "100 uV",
            ha="right",
            va="center",
            rotation=90,
            fontsize=4,
        )
        legend_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        legend_base = output_dir / "unit_waveforms_3c_legend_scalebar"
        legend_fig.savefig(legend_base.with_suffix(".svg"), format="svg", transparent=True)
        legend_fig.savefig(legend_base.with_suffix(".png"), dpi=600, transparent=True)
        legend_fig.savefig(legend_base.with_suffix(".pdf"), format="pdf", transparent=True)
        plt.close(legend_fig)

    with matplotlib.rc_context(
        {
            "path.simplify": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.dash_joinstyle": "round",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for unit_id in unit_ids:
            fig, ax = plt.subplots(figsize=waveform_figsize)
            ax.set_xlim(*waveform_xlim)
            ax.set_ylim(*waveform_ylim)
            ax.axis("off")

            unit_templates = []
            for day_idx, day in enumerate(days):
                matches = np.where(day["unit_ids"] == unit_id)[0]
                if len(matches) == 0:
                    unit_templates.append(None)
                    continue
                unit_index = int(matches[0])
                template = day["templates"][unit_index]
                unit_templates.append(template)

            ordered_channels = plotted_waveform_channels
            center_x = layout_center_x
            center_y = layout_center_y

            for day_idx, template in enumerate(unit_templates):
                if template is None:
                    continue
                for channel_idx in ordered_channels:
                    waveform = template[:, channel_idx]
                    location = channel_locations[channel_idx]
                    t_ms = np.arange(waveform.size) / 30.0
                    plot_x = center_x + (location[0] - center_x) * waveform_map_x_scale
                    plot_y = center_y + (location[1] - center_y) * waveform_map_y_scale
                    x = plot_x + (t_ms - t_ms.mean()) * waveform_time_scale
                    y = plot_y + waveform * waveform_amplitude_scale + (day_idx - 4.5) * day_waveform_y_offset
                    ax.plot(
                        x,
                        y,
                        color=day_colors[day_idx],
                        linewidth=waveform_linewidth,
                        alpha=0.95,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        clip_on=False,
                    )

            frame_color = tracking_colors.get(unit_id, (0.0, 0.0, 0.0, 1.0))
            frame_color = tuple(frame_color[:3]) + (1.0,)
            fig.add_artist(
                Rectangle(
                    (0.015, 0.015),
                    0.97,
                    0.97,
                    transform=fig.transFigure,
                    fill=False,
                    edgecolor=frame_color,
                    linewidth=0.8,
                    linestyle=(0, (7, 5)),
                    zorder=1000,
                )
            )

            fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
            fig.savefig(output_dir / f"Unit{unit_id}_10days_3c.svg", format="svg", transparent=True)
            fig.savefig(output_dir / f"Unit{unit_id}_10days_3c.png", dpi=600, transparent=True)
            fig.savefig(output_dir / f"Unit{unit_id}_10days_3c.pdf", format="pdf", transparent=True)
            plt.close(fig)


def draw_unit_waveform_channel_layout(days: list[dict], output_dir: Path) -> None:
    """Save the channel layout used by the unit waveform panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_locations = days[0]["channel_locations"]
    waveform_layout_channels = sorted(
        {17, 20, 22, 27, 29, 39, 41, 43, 45, 80, 82, 84, 86},
        key=lambda channel: (-channel_locations[channel, 1], channel_locations[channel, 0]),
    )
    waveform_layout_set = set(waveform_layout_channels)
    visible_waveform_set = set(waveform_layout_channels[1:-1])
    hidden_waveform_set = waveform_layout_set - visible_waveform_set
    max_layout_y = max(channel_locations[ch, 1] for ch in waveform_layout_channels)
    min_layout_y = min(channel_locations[ch, 1] for ch in waveform_layout_channels)
    top_channels = sorted(
        [channel for channel in range(len(channel_locations)) if channel_locations[channel, 1] > max_layout_y],
        key=lambda channel: (channel_locations[channel, 1], channel_locations[channel, 0]),
    )[:5]
    bottom_channels = sorted(
        [channel for channel in range(len(channel_locations)) if channel_locations[channel, 1] < min_layout_y],
        key=lambda channel: (-channel_locations[channel, 1], channel_locations[channel, 0]),
    )[:5]
    reference_channels = sorted(
        set(top_channels + waveform_layout_channels + bottom_channels),
        key=lambda channel: (-channel_locations[channel, 1], channel_locations[channel, 0]),
    )
    rows = []
    for channel in reference_channels:
        if channel in visible_waveform_set:
            group = "plotted_layout"
        elif channel in hidden_waveform_set:
            group = "hidden_layout"
        elif channel in set(top_channels):
            group = "top_neighbor"
        else:
            group = "bottom_neighbor"
        rows.append(
            {
                "channel": int(channel),
                "x": float(channel_locations[channel, 0]),
                "y": float(channel_locations[channel, 1]),
                "group": group,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "unit_waveforms_3c_channel_layout.csv", index=False)

    with matplotlib.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(1.65, 2.4))
        ax.set_aspect("equal")
        for channel in reference_channels:
            x, y = channel_locations[channel]
            if channel in visible_waveform_set:
                facecolor = "#d62728"
                edgecolor = "black"
                size = 28
                linewidth = 0.5
            elif channel in hidden_waveform_set:
                facecolor = "#f2a65a"
                edgecolor = "black"
                size = 24
                linewidth = 0.5
            else:
                facecolor = "white"
                edgecolor = "0.45"
                size = 18
                linewidth = 0.4
            ax.scatter(x, y, s=size, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=2)
            ax.text(x + 4.2, y, str(channel), fontsize=5, va="center", ha="left")

        x_vals = channel_locations[reference_channels, 0]
        y_vals = channel_locations[reference_channels, 1]
        ax.set_xlim(float(np.min(x_vals) - 18), float(np.max(x_vals) + 28))
        ax.set_ylim(float(np.min(y_vals) - 30), float(np.max(y_vals) + 30))
        ax.set_xlabel("x", fontsize=6, labelpad=1)
        ax.set_ylabel("y", fontsize=6, labelpad=1)
        ax.tick_params(axis="both", labelsize=5, width=0.6, length=2, pad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_linewidth(0.6)
        fig.subplots_adjust(left=0.25, right=0.96, bottom=0.12, top=0.98)
        out_base = output_dir / "unit_waveforms_3c_channel_layout"
        fig.savefig(out_base.with_suffix(".svg"), format="svg", transparent=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=600, transparent=True)
        fig.savefig(out_base.with_suffix(".pdf"), format="pdf", transparent=True)
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot longitudinal waveforms, tracking, and curated connectivity.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args=parser.parse_args()
    days=load_days(DEFAULT_WAVEFORM_ROOT)
    draw_unit_waveforms(days, args.output_dir / "waveforms", [29, 30, 21, 88, 10])
    draw_unit_waveform_channel_layout(days, args.output_dir / "waveforms")
    draw_tracking_panel(days, args.output_dir / "Fig2F.svg", with_connections=False, figsize=(3.2, 3.2))
    draw_tracking_panel(days, args.output_dir / "Fig2H.svg", with_connections=True, figsize=(2.24, 3.2), wspace=0.25)
