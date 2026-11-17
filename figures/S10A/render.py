"""Reproduce S10A from spatially registered seven-session unit 253 templates."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent

import matplotlib as mpl
import matplotlib.lines as mlines
import pandas as pd

DATA = HERE
STIM_CHANNEL = "raw 121"
BEST_CHANNEL = "raw 117"

def savefig(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", transparent=True, **kwargs)
    plt.close(fig)


def channel_sort_key(row: pd.Series) -> tuple[float, float, str]:
    return (-float(row["y_um"]), float(row["x_um"]), str(row["channel_name"]))


def draw_waveform_neighbors() -> None:
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / "selected_mean_waveforms.npz", allow_pickle=False)
    waveforms = z["mean_waveforms_uV"]
    session_indices = z["session_indices"].astype(int)
    channel_names = [str(name) for name in z["channel_names"]]
    channel_locations = z["channel_locations_um"].astype(float)
    time_ms = z["time_ms"].astype(float)

    rows = pd.DataFrame(
        {
            "channel_name": channel_names,
            "x_um": channel_locations[:, 0],
            "y_um": channel_locations[:, 1],
            "channel_array_index": np.arange(len(channel_names)),
        }
    )
    rows["sort_key"] = rows.apply(channel_sort_key, axis=1)
    rows = rows.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    order = rows["channel_array_index"].to_numpy(int)

    # Match the SF10a raster script: first seven colors from tab20.
    day_colors = np.asarray(plt.get_cmap("tab20").colors[: len(session_indices)])
    center_x = float(np.median(channel_locations[order, 0]))
    center_y = float(np.median(channel_locations[order, 1]))

    # Same design language as current Fig. 2d/old Fig. 3 unit waveform maps:
    # channel groups are expanded from physical coordinates; session traces are
    # vertically offset and color coded.
    waveform_map_x_scale = 2.20 * 1.10 * 1.50
    waveform_map_y_scale = 5.95 * 1.10 * 1.25
    waveform_time_scale = 21.4
    waveform_amplitude_scale = 0.62
    session_waveform_y_offset = 12.0 * 1.50
    waveform_linewidth = 0.3

    plot_centers = []
    for channel_index in order:
        location = channel_locations[channel_index]
        plot_centers.append(
            (
                center_x + (location[0] - center_x) * waveform_map_x_scale,
                center_y + (location[1] - center_y) * waveform_map_y_scale,
            )
        )
    plot_centers = np.asarray(plot_centers)
    xlim = (
        float(plot_centers[:, 0].min() - waveform_time_scale * 1.9),
        float(plot_centers[:, 0].max() + waveform_time_scale * 1.9),
    )
    ylim = (
        float(plot_centers[:, 1].min() - 105),
        float(plot_centers[:, 1].max() + 105),
    )

    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    ):
        fig, ax = plt.subplots(figsize=(1.85, 2.95))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis("off")

        for row_idx, channel_index in enumerate(order):
            channel_name = channel_names[channel_index]
            plot_x, plot_y = plot_centers[row_idx]
            is_stim = channel_name == STIM_CHANNEL
            is_best = channel_name == BEST_CHANNEL
            contact_y = plot_y + 52
            ax.scatter(
                [plot_x],
                [contact_y],
                s=13 if is_stim else 7,
                facecolors="none" if is_stim else "black",
                edgecolors="#d62728" if is_stim else "black",
                linewidths=0.7 if is_stim else 0.25,
                zorder=5,
                clip_on=False,
            )
            if is_stim:
                ax.text(
                    plot_x + waveform_time_scale * 0.7,
                    contact_y + 16,
                    "stim 121",
                    color="#d62728",
                    fontsize=7,
                    ha="left",
                    va="center",
                    clip_on=False,
                )
            elif is_best:
                ax.text(
                    plot_x + waveform_time_scale * 0.7,
                    contact_y + 16,
                    "best 117",
                    color="black",
                    fontsize=7,
                    ha="left",
                    va="center",
                    clip_on=False,
                )

            for session_pos, session_index in enumerate(session_indices):
                waveform = waveforms[session_pos, :, channel_index]
                x = plot_x + (time_ms - time_ms.mean()) * waveform_time_scale
                y = (
                    plot_y
                    + waveform * waveform_amplitude_scale
                    + (session_pos - (len(session_indices) - 1) / 2.0) * session_waveform_y_offset
                )
                ax.plot(
                    x,
                    y,
                    color=day_colors[session_pos],
                    linewidth=waveform_linewidth,
                    alpha=0.95,
                    clip_on=False,
                )

        handles = [
            mlines.Line2D([], [], color=day_colors[i], linewidth=0.8, label=f"S{i + 1}")
            for i in range(len(session_indices))
        ]
        ax.legend(
            handles=handles,
            title="Session",
            loc="lower left",
            frameon=False,
            fontsize=6,
            title_fontsize=7,
            labelspacing=0.45,
            handlelength=1.0,
            handletextpad=0.35,
            borderaxespad=0.0,
        )

        x0 = xlim[0] + 0.60 * (xlim[1] - xlim[0])
        y0 = ylim[0] + 0.10 * (ylim[1] - ylim[0])
        ax.plot([x0, x0 + waveform_time_scale], [y0, y0], color="black", linewidth=0.8)
        ax.plot([x0, x0], [y0, y0 + 100 * waveform_amplitude_scale], color="black", linewidth=0.8)
        ax.text(x0 + waveform_time_scale / 2, y0 - 0.025 * (ylim[1] - ylim[0]), "1 ms", ha="center", va="top", fontsize=6)
        ax.text(
            x0 - 0.04 * (xlim[1] - xlim[0]),
            y0 + 50 * waveform_amplitude_scale,
            "100 uV",
            ha="right",
            va="center",
            rotation=90,
            fontsize=6,
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        out_stem = PUBLICATION / "SF10a_unit253_waveform_neighbors_sessions"
        savefig(fig, out_stem)

        qc = rows.copy()
        qc["is_stim_channel"] = qc["channel_name"] == STIM_CHANNEL
        qc["is_best_channel"] = qc["channel_name"] == BEST_CHANNEL
        qc.to_csv(PUBLICATION / "plot_channel_order.csv", index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    PUBLICATION = parser.parse_args().out
    draw_waveform_neighbors()
