from __future__ import annotations


from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle


DEFAULT_ANCHORS = (99, 76)
DEFAULT_BUBBLE_AREA_SCALE = 0.5
DEFAULT_WAVEFORM_AMPLITUDE_SCALE = 2.5
DEFAULT_FIGURE_HEIGHT_IN = 3.63
DEFAULT_FIGURE_ASPECT = 5.4 / 9.4
DEFAULT_WAVEFORM_YLIM = (-100.0, 50.0)
DEFAULT_WAVEFORM_BOX_Y_SCALE = 1.25
BUBBLE_SIZE_PER_UV = 2.0
BUBBLE_EDGE_COLOR = "black"
BUBBLE_EDGE_WIDTH = 0.5
BUBBLE_ALPHA = 0.3
WAVEFORM_LINE_WIDTH = 0.4
WAVEFORM_ALPHA = 0.95
CONTACT_MARKER_SIZE = 1.5
LEGEND_FONT_SIZE = 6
SCALE_FONT_SIZE = 6
DASHED_CALLOUT_LINE_WIDTH = 1.1

plt.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_inputs(folder):
    with np.load(folder/"map.npz") as data:
        mask = data["approved"]
        waves = [data[f"waveform_{i}"] if ok else np.empty((0,0)) for i,ok in enumerate(data["template_used"])]
        loc = [data[f"location_{i}"] if ok else np.empty(0) for i,ok in enumerate(data["template_used"])]
        return loc, waves, mask, data["contacts"], data["channel_map"]


def estimate_pitch(values: np.ndarray, fallback: float = 10.0) -> float:
    uniq = np.unique(np.sort(values[np.isfinite(values)]))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if diffs.size else fallback


def pick_rows_centered(rows_all: np.ndarray, anchor_row: int, row_max: int) -> np.ndarray:
    rows = np.array(sorted(set(rows_all.tolist() + [anchor_row])))
    if rows.size <= row_max:
        return rows
    anchor_idx = int(np.where(rows == anchor_row)[0][0])
    below = (row_max - 1) // 2
    above = row_max - 1 - below
    lo = max(0, anchor_idx - below)
    hi = min(rows.size, anchor_idx + above + 1)
    if hi - lo < row_max:
        if lo == 0:
            hi = min(rows.size, row_max)
        else:
            lo = max(0, hi - row_max)
    return rows[lo:hi]


def unit_palette(n_units: int, seed: int = 13) -> np.ndarray:
    cmap = plt.get_cmap("hsv")
    colors = cmap(np.linspace(0, 1, n_units, endpoint=False))[:, :3]
    rng = np.random.default_rng(seed)
    return colors[rng.permutation(n_units)]


def size_linear(amp_uv: float, base_level: float = 25.0, base_size: float = 220.0) -> float:
    return max(0.0, float(amp_uv)) / max(np.finfo(float).eps, base_level) * base_size


def bubble_size_figure2(amp_uv: float, scale: float = 1.0) -> float:
    return max(0.0, float(amp_uv)) * BUBBLE_SIZE_PER_UV * scale


def scale_axes_box_from_center(ax: plt.Axes, x_scale: float = 1.0, y_scale: float = 1.0) -> None:
    """Resize an axes box around its center without changing its data coordinates."""
    bbox = ax.get_position()
    center_x = bbox.x0 + bbox.width / 2
    center_y = bbox.y0 + bbox.height / 2
    width = bbox.width * x_scale
    height = bbox.height * y_scale
    ax.set_position([center_x - width / 2, center_y - height / 2, width, height])


def summarize_units(wf_cell, approved, channel_xyz, dat_v_map):
    finite_geom = np.isfinite(channel_xyz[:, 2])
    found = np.isin(dat_v_map, channel_xyz[finite_geom, 2])
    valid_dat = dat_v_map[found]
    id_to_geom = {
        int(ch_id): i
        for i, ch_id in enumerate(channel_xyz[:, 2])
        if np.isfinite(ch_id)
    }
    geom_idx = np.array([id_to_geom[int(ch_id)] for ch_id in valid_dat])
    x = channel_xyz[geom_idx, 0]
    y = channel_xyz[geom_idx, 1]

    top_idx = np.full(len(wf_cell), np.nan)
    top_xy = np.full((len(wf_cell), 2), np.nan)
    top_amp = np.full(len(wf_cell), np.nan)

    for u, wf in enumerate(wf_cell):
        if wf.size == 0:
            continue
        wf_valid = wf[found, :]
        ch_amp = np.nanmax(wf_valid, axis=1) - np.nanmin(wf_valid, axis=1)
        i_top = int(np.nanargmax(ch_amp))
        top_idx[u] = i_top
        top_xy[u] = (x[i_top], y[i_top])
        top_amp[u] = ch_amp[i_top]

    drawable = np.where(approved & np.isfinite(top_amp))[0]
    return found, geom_idx, x, y, top_idx, top_xy, top_amp, drawable


def plot_bubble(
    folder: Path,
    output: Path,
    anchors: tuple[int, int],
    annotate: bool = True,
    bubble_area_scale: float = DEFAULT_BUBBLE_AREA_SCALE,
    waveform_amplitude_scale: float = DEFAULT_WAVEFORM_AMPLITUDE_SCALE,
    solid_contacts: bool = False,
    panel_contact_dots: bool = False,
    panel_bubbles: bool = False,
    waveform_scale_bar: bool = True,
    waveform_ylim: tuple[float, float] = DEFAULT_WAVEFORM_YLIM,
    waveform_box_y_scale: float = DEFAULT_WAVEFORM_BOX_Y_SCALE,
    approved_override: np.ndarray | None = None,
    figure5_panel_c_style: bool = False,
    input_data: tuple | None = None,
):
    loc_cell, wf_cell, approved, channel_xyz, dat_v_map = (
        load_inputs(folder) if input_data is None else input_data
    )
    if approved_override is not None:
        approved_override = np.asarray(approved_override, dtype=bool).ravel()
        if approved_override.shape != approved.shape:
            raise ValueError(
                f"approved_override has shape {approved_override.shape}, expected {approved.shape}"
            )
        approved = approved_override
    found, geom_idx, x, y, top_idx, top_xy, top_amp, drawable = summarize_units(
        wf_cell, approved, channel_xyz, dat_v_map
    )

    x_all = channel_xyz[:, 0]
    y_all = channel_xyz[:, 1]
    colors = unit_palette(len(wf_cell))

    fig = plt.figure(
        figsize=(DEFAULT_FIGURE_HEIGHT_IN * DEFAULT_FIGURE_ASPECT, DEFAULT_FIGURE_HEIGHT_IN),
        facecolor="white",
        constrained_layout=False,
    )
    ax_left = fig.add_axes([0.09, 0.04, 0.22, 0.92])
    ax_left.scatter(
        x_all,
        y_all,
        s=CONTACT_MARKER_SIZE if solid_contacts else 7,
        marker="o" if solid_contacts else "s",
        c="black" if solid_contacts else "0.55",
        alpha=0.90 if solid_contacts else 0.20,
        edgecolors="none",
        clip_on=False,
    )

    draw_order = drawable[np.argsort(top_amp[drawable])[::-1]]
    for u in draw_order:
        loc = np.asarray(loc_cell[u])
        if loc.size != 2 or not np.all(np.isfinite(loc)):
            continue
        ax_left.scatter(
            loc[0],
            loc[1],
            s=bubble_size_figure2(top_amp[u], bubble_area_scale),
            facecolors=[colors[u]],
            alpha=BUBBLE_ALPHA,
            edgecolors=BUBBLE_EDGE_COLOR,
            linewidths=BUBBLE_EDGE_WIDTH,
            clip_on=False,
        )

    xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
    ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
    ax_left.set_aspect("equal")
    ax_left.set_xlim(xmin - 90, xmax + 86)
    ax_left.set_ylim(ymin - 28, ymax + 28)
    ax_left.axis("off")

    legend_rect = [0.370, 0.925, 0.500, 0.045] if figure5_panel_c_style else None
    add_bubble_legend(
        fig,
        bubble_area_scale,
        legend_rect=legend_rect,
        orientation="horizontal" if figure5_panel_c_style else "vertical",
    )

    panel_rects = (
        ([0.385, 0.535, 0.535, 0.375], [0.385, 0.075, 0.535, 0.360])
        if figure5_panel_c_style
        else ([0.40, 0.55, 0.53, 0.40], [0.40, 0.06, 0.53, 0.38])
    )
    panel_axes = []
    for panel_i, anchor_unit in enumerate(anchors):
        panel_axes_here = draw_overlay_panel(
            fig,
            panel_rects[panel_i],
            anchor_unit - 1,
            wf_cell,
            found,
            x_all,
            y_all,
            channel_xyz,
            dat_v_map,
            loc_cell,
            top_idx,
            top_xy,
            top_amp,
            colors,
            bubble_area_scale=bubble_area_scale,
            waveform_amplitude_scale=waveform_amplitude_scale,
            panel_contact_dots=panel_contact_dots,
            panel_bubbles=panel_bubbles,
            waveform_ylim=waveform_ylim,
            waveform_box_y_scale=waveform_box_y_scale,
        )
        panel_axes.append(panel_axes_here)

    if waveform_scale_bar:
        reference_axes = next((axes for axes in reversed(panel_axes) if axes), [])
        if reference_axes:
            add_waveform_scale_bar(
                fig,
                reference_axes[0],
                waveform_amplitude_scale=waveform_amplitude_scale,
                panel_rect=panel_rects[-1],
            )

    if annotate:
        add_callouts(fig, ax_left, panel_rects, anchors, loc_cell, filled_arrowhead=figure5_panel_c_style)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(fig)
    return output


def add_bubble_legend(fig, bubble_area_scale, legend_rect=None, orientation="vertical"):
    """Draw the amplitude bubble legend outside the equal-aspect probe axes."""
    if legend_rect is None:
        legend_rect = [0.010, 0.245, 0.075, 0.115]
    legend_ax = fig.add_axes(legend_rect, frameon=False)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    levels = (100, 50, 25)
    if orientation == "horizontal":
        positions = [(0.05, 0.50), (0.45, 0.50), (0.73, 0.50)]
        text_dx = 0.065
    else:
        positions = [(0.48, 0.78), (0.48, 0.48), (0.48, 0.22)]
        text_dx = 0.22
    for level, (xx, yy) in zip(levels, positions):
        legend_ax.scatter(
            xx,
            yy,
            s=bubble_size_figure2(level, bubble_area_scale),
            facecolors="0.75",
            alpha=BUBBLE_ALPHA,
            edgecolors=BUBBLE_EDGE_COLOR,
            linewidths=BUBBLE_EDGE_WIDTH,
            clip_on=False,
        )
        legend_ax.text(xx + text_dx, yy, f"{level}\u03bcV", va="center", ha="left", fontsize=LEGEND_FONT_SIZE)


def draw_overlay_panel(
    fig,
    rect,
    anchor,
    wf_cell,
    found,
    x_all,
    y_all,
    channel_xyz,
    dat_v_map,
    loc_cell,
    top_idx,
    top_xy,
    top_amp,
    colors,
    fs=30000,
    unit_radius_um=60,
    channel_y_window_um=50,
    row_max=10,
    bubble_area_scale=DEFAULT_BUBBLE_AREA_SCALE,
    waveform_amplitude_scale=DEFAULT_WAVEFORM_AMPLITUDE_SCALE,
    panel_contact_dots=False,
    panel_bubbles=False,
    waveform_ylim=DEFAULT_WAVEFORM_YLIM,
    waveform_box_y_scale=DEFAULT_WAVEFORM_BOX_Y_SCALE,
):
    if anchor < 0 or anchor >= len(wf_cell) or not np.isfinite(top_idx[anchor]):
        return []

    dist = np.hypot(top_xy[:, 0] - top_xy[anchor, 0], top_xy[:, 1] - top_xy[anchor, 1])
    units_here = np.where(dist <= unit_radius_um)[0]
    idx_ch = np.where(np.abs(y_all - top_xy[anchor, 1]) <= channel_y_window_um)[0]
    if idx_ch.size == 0 or units_here.size == 0:
        return []

    plot_rect = [
        rect[0] + 0.035,
        rect[1] + 0.045,
        rect[2] - 0.070,
        rect[3] - 0.095,
    ]

    pitch_y = estimate_pitch(y_all)
    yq = np.round((y_all - y_all[idx_ch].min()) / max(np.finfo(float).eps, pitch_y)).astype(int)
    anchor_row = int(round((top_xy[anchor, 1] - y_all[idx_ch].min()) / max(np.finfo(float).eps, pitch_y)))
    rows_sel = pick_rows_centered(np.unique(yq[idx_ch]), anchor_row, row_max)
    row_h = plot_rect[3] / max(1, len(rows_sel))
    ax_h = row_h * 0.80
    center_row = int(np.ceil((len(rows_sel) + 1) / 2))

    x_local = x_all[idx_ch]
    x_min, x_max = x_local.min(), x_local.max()
    den_x = max(np.finfo(float).eps, x_max - x_min)
    n_cols = max(3, len(np.unique(np.round((x_local - x_min) / max(np.finfo(float).eps, estimate_pitch(x_local))))))
    ax_w = (plot_rect[2] / n_cols) * 0.70

    id_to_valid = {int(ch_id): i for i, ch_id in enumerate(dat_v_map[found])}
    time_ms = np.arange(wf_cell[0].shape[1]) / fs * 1000

    y_min, y_max = np.inf, -np.inf
    for ch in idx_ch:
        rv = id_to_valid.get(int(channel_xyz[ch, 2]))
        if rv is None:
            continue
        for unit in units_here:
            yy = wf_cell[unit][found, :][rv, :]
            y_min = min(y_min, float(np.nanmin(yy)))
            y_max = max(y_max, float(np.nanmax(yy)))
    if not np.isfinite(y_min):
        y_min, y_max = -100, 100
    pad = 0.08 * max(1.0, y_max - y_min)

    axes = []
    for row in rows_sel:
        ch_row = idx_ch[yq[idx_ch] == row]
        ch_row = ch_row[np.argsort(x_all[ch_row])]
        for ch in ch_row:
            rv = id_to_valid.get(int(channel_xyz[ch, 2]))
            if rv is None:
                continue
            row_offset = np.where(rows_sel == row)[0][0] - np.where(rows_sel == anchor_row)[0][0]
            display_row = center_row + row_offset
            left = plot_rect[0] + ((x_all[ch] - x_min) / den_x) * (plot_rect[2] - ax_w)
            bottom = plot_rect[1] + (display_row - 1) * row_h + (row_h - ax_h) / 2
            ax = fig.add_axes([left, bottom, ax_w, ax_h])
            ax.set_zorder(3)
            ax.patch.set_alpha(0)
            ax.axis("off")
            ax.set_ylim(*waveform_ylim)
            scale_axes_box_from_center(ax, y_scale=waveform_box_y_scale)
            amps = []
            plot_units = []
            for unit in units_here:
                yy = wf_cell[unit][found, :][rv, :]
                amps.append(np.nanmax(yy) - np.nanmin(yy))
                plot_units.append(unit)
            if panel_contact_dots:
                ax.scatter(
                    [0.50],
                    [1.05],
                    s=CONTACT_MARKER_SIZE,
                    c="black",
                    marker="o",
                    edgecolors="none",
                    transform=ax.transAxes,
                    clip_on=False,
                    zorder=6,
                )
            for unit in np.array(plot_units)[np.argsort(amps)]:
                yy = wf_cell[unit][found, :][rv, :]
                ax.plot(
                    time_ms,
                    yy * waveform_amplitude_scale,
                    color=colors[unit],
                    lw=WAVEFORM_LINE_WIDTH,
                    alpha=WAVEFORM_ALPHA,
                    clip_on=False,
                    zorder=4,
                )
            ax.set_xlim(time_ms[0], 2 - 1 / 3)
            axes.append(ax)
    if panel_bubbles:
        add_panel_unit_bubbles(
            fig,
            plot_rect,
            loc_cell,
            units_here,
            top_amp,
            colors,
            x_min,
            x_max,
            y_all[idx_ch].min(),
            y_all[idx_ch].max(),
            bubble_area_scale,
        )
    return axes


def add_panel_unit_bubbles(
    fig,
    rect,
    loc_cell,
    units_here,
    top_amp,
    colors,
    x_min,
    x_max,
    y_min,
    y_max,
    bubble_area_scale,
):
    """Add Figure 2-style unit bubbles on a transparent geometry overlay."""
    bubble_ax = fig.add_axes(rect, frameon=False)
    bubble_ax.set_zorder(1)
    bubble_ax.patch.set_alpha(0)
    bubble_ax.set_xlim(x_min, x_max)
    bubble_ax.set_ylim(y_min, y_max)
    bubble_ax.set_xticks([])
    bubble_ax.set_yticks([])
    bubble_ax.axis("off")

    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    bubble_ax.set_xlim(x_min - 0.35 * x_span, x_max + 0.35 * x_span)
    bubble_ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.10 * y_span)

    drawable = []
    for unit in units_here:
        loc = np.asarray(loc_cell[unit])
        if loc.size == 2 and np.all(np.isfinite(loc)) and np.isfinite(top_amp[unit]):
            drawable.append(unit)
    drawable = np.array(drawable, dtype=int)
    if drawable.size == 0:
        return

    for unit in drawable[np.argsort(top_amp[drawable])[::-1]]:
        loc = np.asarray(loc_cell[unit])
        bubble_ax.scatter(
            loc[0],
            loc[1],
            s=bubble_size_figure2(top_amp[unit], bubble_area_scale),
            facecolors=[colors[unit]],
            alpha=BUBBLE_ALPHA,
            edgecolors=BUBBLE_EDGE_COLOR,
            linewidths=BUBBLE_EDGE_WIDTH,
            clip_on=False,
            zorder=1,
        )


def add_waveform_scale_bar(
    fig,
    reference_ax,
    waveform_amplitude_scale=DEFAULT_WAVEFORM_AMPLITUDE_SCALE,
    panel_rect=None,
    scale_time_ms=1.0,
    scale_voltage_uv=50.0,
):
    """Draw a Figure 2-style waveform scale bar in figure coordinates."""
    bbox = reference_ax.get_position()
    xlim = reference_ax.get_xlim()
    ylim = reference_ax.get_ylim()
    bar_width = bbox.width * scale_time_ms / max(np.finfo(float).eps, xlim[1] - xlim[0])
    bar_height = bbox.height * (scale_voltage_uv * waveform_amplitude_scale) / max(
        np.finfo(float).eps,
        ylim[1] - ylim[0],
    )

    if panel_rect is None:
        x0 = bbox.x0
        y0 = bbox.y0 - 0.025
    else:
        x0 = panel_rect[0] - 0.035
        y0 = panel_rect[1] - 0.035

    fig.add_artist(Line2D([x0, x0 + bar_width], [y0, y0], transform=fig.transFigure, color="black", lw=1.2))
    fig.add_artist(Line2D([x0, x0], [y0, y0 + bar_height], transform=fig.transFigure, color="black", lw=1.2))
    fig.text(x0 + bar_width / 2, y0 - 0.010, "1 ms", ha="center", va="top", fontsize=SCALE_FONT_SIZE)
    fig.text(
        x0 - 0.015,
        y0 + bar_height / 2,
        f"{scale_voltage_uv:g} uV",
        ha="right",
        va="center",
        rotation=90,
        fontsize=SCALE_FONT_SIZE,
    )


def add_callouts(fig, ax_left, panel_rects, anchors, loc_cell, filled_arrowhead=False):
    callout_colors = ("red", "#0072B2")
    for i, anchor_unit in enumerate(anchors):
        color = callout_colors[i]
        loc = np.asarray(loc_cell[anchor_unit - 1])
        if loc.size == 2 and np.all(np.isfinite(loc)):
            box_w, box_h = 54, 88
            ax_left.add_patch(
                Rectangle(
                    (loc[0] - box_w / 2, loc[1] - box_h / 2),
                    box_w,
                    box_h,
                    fill=False,
                    ec=color,
                    lw=DASHED_CALLOUT_LINE_WIDTH,
                    ls=(0, (3, 3)),
                )
            )

            target = (panel_rects[i][0], panel_rects[i][1] + panel_rects[i][3] * 0.68)
            fig.add_artist(
                ConnectionPatch(
                    xyA=(loc[0] + box_w / 2, loc[1]),
                    coordsA=ax_left.transData,
                    xyB=target,
                    coordsB=fig.transFigure,
                    arrowstyle="-|>" if filled_arrowhead else "->",
                    mutation_scale=8 if filled_arrowhead else 16,
                    lw=DASHED_CALLOUT_LINE_WIDTH,
                    ls=(0, (3, 3)),
                    color=color,
                    fc=color,
                    ec=color,
                )
            )

        rect = panel_rects[i]
        pad = 0.008 if filled_arrowhead else 0.012
        fig.add_artist(
            Rectangle(
                (rect[0] - pad, rect[1] - pad),
                rect[2] + 2 * pad,
                rect[3] + 2 * pad,
                transform=fig.transFigure,
                fill=False,
                ec=color,
                lw=DASHED_CALLOUT_LINE_WIDTH,
                ls=(0, (3, 3)),
            )
        )


