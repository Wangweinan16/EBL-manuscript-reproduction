#!/usr/bin/env python3
"""Publication-style panels for corrected large-sweep in-plane/out-plane runs."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, Literal

import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent
INPLANE_CSV = ROOT / "ltspice_results_from_comsol_large_sweep_1em9_inplane_corrected_pi" / "ltspice_final_xtalk_results.csv"
OUTPLANE_CSV = ROOT / "ltspice_results_from_comsol_large_sweep_1em9_outplane" / "ltspice_final_xtalk_results.csv"
DEFAULT_OUTDIR = ROOT / "publishable_large_sweep_1em9_corrected_pi"
GAP_RATIOS = (0.2, 0.5, 1.0, 2.0, 4.0)
DEFAULT_RSITE_TARGETS = (100e3, 2e6)
PI_TARGET_UM = 1.9
YScale = Literal["linear", "log"]
INPLANE_ST_INSET_XLIM = (1e4, 1e5)


@dataclass(frozen=True)
class FigureStyle:
    panel_size: tuple[float, float] = (2.4, 2.0)
    dpi: int = 600
    font_size: float = 6.0
    label_size: float = 6.5
    title_size: float = 7.0
    tick_size: float = 5.5
    legend_size: float = 5.5
    line_width: float = 1.0
    marker_size: float = 2.0


def parse_size(text: str, default: tuple[float, float]) -> tuple[float, float]:
    if not text:
        return default
    parts = [p.strip() for p in text.lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected WIDTH,HEIGHT in inches, got {text!r}")
    return float(parts[0]), float(parts[1])


def apply_style(style: FigureStyle) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": style.font_size,
            "axes.labelsize": style.label_size,
            "axes.titlesize": style.title_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.legend_size,
            "axes.linewidth": 0.6,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": style.dpi,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    mpl.rcParams["svg.fonttype"] = "path"


def require_csv(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def nearest(values: pd.Series, target: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return float(vals.iloc[(vals - target).abs().argmin()])


def close_to(values: pd.Series, target: float) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    return (vals - target).abs() <= max(abs(target) * 1e-9, 1e-30)


def select_rows(df: pd.DataFrame, rsite_target: float) -> tuple[pd.DataFrame, dict[str, float]]:
    selected = df.copy()
    actual_pi = nearest(selected["pi_t_um"], PI_TARGET_UM)
    selected = selected.loc[close_to(selected["pi_t_um"], actual_pi)].copy()
    actual_rsite = nearest(selected["ltspice_r_site_ohm"], rsite_target)
    selected = selected.loc[close_to(selected["ltspice_r_site_ohm"], actual_rsite)].copy()
    selected = selected.loc[close_to(selected["frequency_hz"], nearest(selected["frequency_hz"], 1000.0))].copy()
    selected = selected[selected["gap_ratio"].isin(GAP_RATIOS)].copy()
    return selected.sort_values(["gap_ratio", "width_nm"]).reset_index(drop=True), {
        "pi_t_um": actual_pi,
        "rsite_ohm": actual_rsite,
    }


def format_rsite(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g} Mohm"
    if value >= 1e3:
        return f"{value / 1e3:g} kohm"
    return f"{value:g} ohm"


def save_figure(fig: plt.Figure, outbase: pathlib.Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(outbase.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03)


def padded_ylim(values: Iterable[float], lower: float = 0.0) -> tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return lower, 1.0
    ymax = float(vals.max())
    if ymax <= lower:
        ymax = lower + 1.0
    return lower, ymax * 1.08


def display_curve(curve: pd.DataFrame, x_col: str, y_col: str, *, points: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate y vs log10(x) for smoother display curves; source data are unchanged."""
    x = pd.to_numeric(curve[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(curve[y_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    log_x = np.log10(x)
    log_x_dense = np.linspace(float(log_x.min()), float(log_x.max()), points)
    return np.power(10.0, log_x_dense), np.interp(log_x_dense, log_x, y)


def plot_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    style: FigureStyle,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    yscale: YScale = "linear",
    xlim: tuple[float, float] | None = None,
    interpolate: bool = False,
) -> None:
    for gap in GAP_RATIOS:
        curve = df.loc[close_to(df["gap_ratio"], gap)].sort_values("width_nm")
        if curve.empty:
            continue
        if interpolate:
            x, y = display_curve(curve, "width_nm", y_col)
        else:
            x, y = curve["width_nm"], curve[y_col]
        ax.semilogx(
            x,
            y,
            marker=None,
            linewidth=style.line_width,
            label=f"{gap:g}",
        )
    ax.set_title(title)
    ax.set_xlabel("Line width (nm)")
    ax.set_ylabel(ylabel)
    if yscale == "log":
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False)
    ax.tick_params(length=2.5)


def autoscale_y_with_padding(ax: plt.Axes, *, pad_frac: float = 0.1, min_pad: float = 1e-6) -> None:
    ax.autoscale(axis="y")
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        return
    pad = max((ymax - ymin) * pad_frac, min_pad)
    ax.set_ylim(ymin - pad, ymax + pad)


def save_single_panel(
    outdir: pathlib.Path,
    basename: str,
    df: pd.DataFrame,
    y_col: str,
    style: FigureStyle,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    yscale: YScale,
    interpolate: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=style.panel_size, constrained_layout=True)
    plot_metric_panel(
        ax,
        df,
        y_col,
        style,
        title=title,
        ylabel=ylabel,
        ylim=ylim,
        yscale=yscale,
        interpolate=interpolate,
    )
    ax.legend(title="Gap / Line width", frameon=False)
    save_figure(fig, outdir / "figures" / basename)
    plt.close(fig)


def filter_xrange(df: pd.DataFrame, xlim: tuple[float, float]) -> pd.DataFrame:
    x = pd.to_numeric(df["width_nm"], errors="coerce")
    return df.loc[(x >= xlim[0]) & (x <= xlim[1])].copy()


def save_inset_panel(
    outdir: pathlib.Path,
    basename: str,
    df: pd.DataFrame,
    y_col: str,
    style: FigureStyle,
    *,
    title: str,
    ylabel: str,
    inset_xlim: tuple[float, float],
    interpolate: bool = False,
) -> None:
    """Standalone zoom panel (log x, linear y); y limits autoscale to data in xlim."""
    zoom_df = filter_xrange(df, inset_xlim)
    inset_style = FigureStyle(
        panel_size=(style.panel_size[0] * 2.0 / 3.0, style.panel_size[1] * 2.0 / 3.0),
        dpi=style.dpi,
        font_size=style.font_size * 0.72,
        label_size=style.label_size * 0.72,
        title_size=style.title_size * 0.72,
        tick_size=style.tick_size * 0.72,
        legend_size=style.legend_size * 0.72,
        line_width=style.line_width,
        marker_size=style.marker_size,
    )
    fig, ax = plt.subplots(figsize=inset_style.panel_size, constrained_layout=True)
    plot_metric_panel(
        ax,
        zoom_df,
        y_col,
        inset_style,
        title="",
        ylabel="",
        ylim=None,
        yscale="linear",
        xlim=inset_xlim,
        interpolate=interpolate,
    )
    autoscale_y_with_padding(ax)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.FixedLocator([inset_xlim[0], inset_xlim[1]]))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    save_figure(fig, outdir / "figures" / basename)
    plt.close(fig)


def make_figures(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    outdir: pathlib.Path,
    *,
    style: FigureStyle,
    rsite_targets: tuple[float, ...],
    inplane_csv: pathlib.Path,
    outplane_csv: pathlib.Path,
) -> None:
    apply_style(style)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figure_data").mkdir(exist_ok=True)

    selected: dict[float, dict[str, pd.DataFrame]] = {}
    actuals: dict[float, dict[str, float]] = {}
    for rsite in rsite_targets:
        in_sel, in_actual = select_rows(in_df, rsite)
        out_sel, out_actual = select_rows(out_df, rsite)
        selected[rsite] = {"inplane": in_sel, "outplane": out_sel}
        actuals[rsite] = {
            "in_rsite": in_actual["rsite_ohm"],
            "out_rsite": out_actual["rsite_ohm"],
            "pi": in_actual["pi_t_um"],
        }
        in_sel.to_csv(outdir / "figure_data" / f"inplane_rsite_{int(in_actual['rsite_ohm'])}.csv", index=False)
        out_sel.to_csv(outdir / "figure_data" / f"outplane_rsite_{int(out_actual['rsite_ohm'])}.csv", index=False)

    all_xtalk_lin: list[float] = []
    all_signal_lin: list[float] = []
    for rsite in rsite_targets:
        for df in selected[rsite].values():
            xtalk = df["xtalk_pct"].to_numpy(dtype=float)
            signal = df["aggressor_gain_mag"].to_numpy(dtype=float)
            all_xtalk_lin.extend(xtalk)
            all_signal_lin.extend(signal)

    xtalk_ylim_lin = padded_ylim(all_xtalk_lin)
    signal_ylim_lin = padded_ylim(all_signal_lin)

    rsite_list = ", ".join(format_rsite(r) for r in rsite_targets)
    for rsite in rsite_targets:
        in_sel = selected[rsite]["inplane"]
        out_sel = selected[rsite]["outplane"]
        actual_rsite = actuals[rsite]["in_rsite"]
        pi_title = "PI thickness = 2 um"
        rsite_title = f"Zsite = {format_rsite(actual_rsite)}"
        rsite_tag = int(actual_rsite)

        fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), constrained_layout=True, sharex=True)
        plot_metric_panel(
            axes[0, 0],
            in_sel,
            "xtalk_pct",
            style,
            title=f"In-plane cross talk\n{pi_title}, {rsite_title}",
            ylabel="Cross talk (%)",
            ylim=xtalk_ylim_lin,
        )
        plot_metric_panel(
            axes[0, 1],
            out_sel,
            "xtalk_pct",
            style,
            title=f"Out-plane cross talk\n{pi_title}, {rsite_title}",
            ylabel="Cross talk (%)",
            ylim=xtalk_ylim_lin,
        )
        plot_metric_panel(
            axes[1, 0],
            in_sel,
            "aggressor_gain_mag",
            style,
            title=f"In-plane signal transfer\n{pi_title}, {rsite_title}",
            ylabel="Signal transfer (V/V)",
            ylim=signal_ylim_lin,
        )
        plot_metric_panel(
            axes[1, 1],
            out_sel,
            "aggressor_gain_mag",
            style,
            title=f"Out-plane signal transfer\n{pi_title}, {rsite_title}",
            ylabel="Signal transfer (V/V)",
            ylim=(0.4, signal_ylim_lin[1]),
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Gap / Line width",
            loc="outside lower center",
            ncol=len(GAP_RATIOS),
            frameon=False,
        )
        save_figure(fig, outdir / "figures" / f"large_sweep_4panel_rsite_{rsite_tag}ohm")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), constrained_layout=True, sharex=True)
        plot_metric_panel(
            axes[0, 0],
            in_sel,
            "xtalk_pct",
            style,
            title=f"In-plane cross talk\n{pi_title}, {rsite_title}",
            ylabel="Cross talk (%)",
            ylim=xtalk_ylim_lin,
            interpolate=True,
        )
        plot_metric_panel(
            axes[0, 1],
            out_sel,
            "xtalk_pct",
            style,
            title=f"Out-plane cross talk\n{pi_title}, {rsite_title}",
            ylabel="Cross talk (%)",
            ylim=xtalk_ylim_lin,
            interpolate=True,
        )
        plot_metric_panel(
            axes[1, 0],
            in_sel,
            "aggressor_gain_mag",
            style,
            title=f"In-plane signal transfer\n{pi_title}, {rsite_title}",
            ylabel="Signal transfer (V/V)",
            ylim=signal_ylim_lin,
            interpolate=True,
        )
        plot_metric_panel(
            axes[1, 1],
            out_sel,
            "aggressor_gain_mag",
            style,
            title=f"Out-plane signal transfer\n{pi_title}, {rsite_title}",
            ylabel="Signal transfer (V/V)",
            ylim=(0.4, signal_ylim_lin[1]),
            interpolate=True,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Gap / Line width",
            loc="outside lower center",
            ncol=len(GAP_RATIOS),
            frameon=False,
        )
        save_figure(fig, outdir / "figures" / f"large_sweep_4panel_rsite_{rsite_tag}ohm_interp")
        plt.close(fig)

        panel_specs = [
            (
                "inplane_xtalk",
                in_sel,
                "xtalk_pct",
                "In-plane cross talk",
                "Cross talk (%)",
                xtalk_ylim_lin,
            ),
            (
                "outplane_xtalk",
                out_sel,
                "xtalk_pct",
                "Out-plane cross talk",
                "Cross talk (%)",
                xtalk_ylim_lin,
            ),
            (
                "inplane_signal_transfer",
                in_sel,
                "aggressor_gain_mag",
                "In-plane signal transfer",
                "Signal transfer (V/V)",
                signal_ylim_lin,
            ),
            (
                "outplane_signal_transfer",
                out_sel,
                "aggressor_gain_mag",
                "Out-plane signal transfer",
                "Signal transfer (V/V)",
                (0.4, signal_ylim_lin[1]),
            ),
        ]
        for name, df, y_col, title, ylabel, ylim_lin in panel_specs:
            panel_title = f"{title}\n{pi_title}, {rsite_title}"
            base = f"{name}_rsite_{rsite_tag}ohm"
            save_single_panel(
                outdir,
                base,
                df,
                y_col,
                style,
                title=panel_title,
                ylabel=ylabel,
                ylim=ylim_lin,
                yscale="linear",
            )
            save_single_panel(
                outdir,
                f"{base}_interp",
                df,
                y_col,
                style,
                title=panel_title,
                ylabel=ylabel,
                ylim=ylim_lin,
                yscale="linear",
                interpolate=True,
            )
            if name == "inplane_signal_transfer":
                x0, x1 = INPLANE_ST_INSET_XLIM
                save_inset_panel(
                    outdir,
                    f"{base}_inset",
                    df,
                    y_col,
                    style,
                    title=f"{panel_title}\nLine width zoom {x0:g}–{x1:g} nm",
                    ylabel=ylabel,
                    inset_xlim=INPLANE_ST_INSET_XLIM,
                )
                save_inset_panel(
                    outdir,
                    f"{base}_inset_interp",
                    df,
                    y_col,
                    style,
                    title=f"{panel_title}\nLine width zoom {x0:g}–{x1:g} nm",
                    ylabel=ylabel,
                    inset_xlim=INPLANE_ST_INSET_XLIM,
                    interpolate=True,
                )

    readme = f"""# Publishable Large-Sweep Panels

Style aligned with `publishable_xtalk_figures/figures_v2` (panel size {style.panel_size[0]} x {style.panel_size[1]} in, line-only curves, no markers).

Inputs:

- In-plane: `{inplane_csv}`
- Out-plane: `{outplane_csv}`

Selections:

- PI thickness uses corrected pipeline `pi_t = 1.9 um`, displayed in titles as `PI thickness = 2 um`.
- Site impedances requested: {rsite_list}.
- Gap-ratio curves shown under legend title `Gap / Line width`: `0.2, 0.5, 1, 2, 4`.
- X axis: line width in nm, log scaled.
- Linear-y cross-talk panels share one y-axis range across plane orientations.
- Linear-y signal-transfer panels use a 0 lower limit except out-plane signal transfer, which starts at 0.4.

Outputs:

- `figures/large_sweep_4panel_rsite_<ohm>` composite (linear y, PNG/SVG).
- Per-panel linear-y: `inplane_xtalk_rsite_*`, `outplane_xtalk_rsite_*`, `inplane_signal_transfer_rsite_*`, `outplane_signal_transfer_rsite_*`.
- `inplane_signal_transfer_rsite_*_inset`: separate 2/3-size zoom panel for line width `{INPLANE_ST_INSET_XLIM[0]:g}`–`{INPLANE_ST_INSET_XLIM[1]:g}` nm (log x, linear y, autoscale y).
- Files ending in `_interp` use a display-only interpolation of each curve vs `log10(line width)` for smoother rendered lines. Source data in `figure_data/` are unchanged.
- Source rows in `figure_data/`.
"""
    pass  # Source tables and graphics are written above.


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create publication-style large-sweep panels.")
    parser.add_argument("--inplane-csv", type=pathlib.Path, default=INPLANE_CSV)
    parser.add_argument("--outplane-csv", type=pathlib.Path, default=OUTPLANE_CSV)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--rsite-ohm",
        action="append",
        type=float,
        dest="rsite_ohm",
        help="Site impedance target in ohm (repeatable). Default: 100 kOhm and 2 MOhm.",
    )
    parser.add_argument("--panel-size", default="2.4,2.0", help="Single-panel size in inches as WIDTH,HEIGHT")
    parser.add_argument("--font-size", type=float, default=6.0)
    parser.add_argument("--label-size", type=float, default=6.5)
    parser.add_argument("--title-size", type=float, default=7.0)
    parser.add_argument("--tick-size", type=float, default=5.5)
    parser.add_argument("--legend-size", type=float, default=5.5)
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rsite_targets = tuple(args.rsite_ohm) if args.rsite_ohm else DEFAULT_RSITE_TARGETS
    style = FigureStyle(
        panel_size=parse_size(args.panel_size, FigureStyle.panel_size),
        dpi=args.dpi,
        font_size=args.font_size,
        label_size=args.label_size,
        title_size=args.title_size,
        tick_size=args.tick_size,
        legend_size=args.legend_size,
    )
    columns = [
        "width_nm",
        "gap_ratio",
        "pi_t_um",
        "ltspice_r_site_ohm",
        "frequency_hz",
        "xtalk_pct",
        "aggressor_gain_mag",
    ]
    in_df = numeric(require_csv(args.inplane_csv), columns)
    out_df = numeric(require_csv(args.outplane_csv), columns)
    make_figures(
        in_df,
        out_df,
        args.outdir,
        style=style,
        rsite_targets=rsite_targets,
        inplane_csv=args.inplane_csv,
        outplane_csv=args.outplane_csv,
    )
    print(f"Saved publishable panels to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
