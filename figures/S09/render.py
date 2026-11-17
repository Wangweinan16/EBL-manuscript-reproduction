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

import openpyxl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
S11_WORKBOOK = HERE / "neighbor_pair_workbook.xlsx"
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)


PAIR_CONFIGS = [
    {
        "pair_index": 1,
        "dataset": "Feb-15-EBL2",
        "sheet": "Feb15",
        "channel_a": 13,
        "channel_b": 84,
        "source_a": ("L", 18, 23),
        "source_b": ("N", 18, 57),
        "edge_a": (19, 21),
        "edge_b": (23, 42),
    },
    {
        "pair_index": 2,
        "dataset": "Feb-15-EBL2",
        "sheet": "Feb15",
        "channel_a": 94,
        "channel_b": 117,
        "source_a": ("S", 18, 33),
        "source_b": ("U", 18, 69),
        "edge_a": (44, 51),
        "edge_b": (54, 79),
    },
    {
        "pair_index": 3,
        "dataset": "Jan-31-EBL2",
        "sheet": "Jan31",
        "channel_a": 1,
        "channel_b": 37,
        "source_a": ("J", 2, 7),
        "source_b": ("K", 2, 9),
        "edge_a": (2, 4),
        "edge_b": (21, 24),
    },
    {
        "pair_index": 4,
        "dataset": "Jan-31-EBL2",
        "sheet": "Jan31",
        "channel_a": 69,
        "channel_b": 76,
        "source_a": ("J", 12, 49),
        "source_b": ("K", 12, 31),
        "edge_a": (25, 43),
        "edge_b": (44, 53),
    },
    {
        "pair_index": 5,
        "dataset": "Jan-31-EBL2",
        "sheet": "Jan31",
        "channel_a": 11,
        "channel_b": 84,
        "source_a": ("Q", 2, 9),
        "source_b": ("R", 2, 27),
        "edge_a": (5, 8),
        "edge_b": (54, 66),
    },
    {
        "pair_index": 6,
        "dataset": "Jan-31-EBL2",
        "sheet": "Jan31",
        "channel_a": 15,
        "channel_b": 94,
        "source_a": ("Q", 30, 53),
        "source_b": ("R", 30, 65),
        "edge_a": (9, 20),
        "edge_b": (67, 84),
    },
    {
        "pair_index": 7,
        "dataset": "EBL6",
        "sheet": "EBL6",
        "channel_a": 70,
        "channel_b": 99,
        "source_a": ("G", 2, 17),
        "source_b": ("H", 2, 17),
        "edge_a": (2, 9),
        "edge_b": (20, 27),
    },
    {
        "pair_index": 8,
        "dataset": "EBL6",
        "sheet": "EBL6",
        "channel_a": 72,
        "channel_b": 106,
        "source_a": ("N", 2, 21),
        "source_b": ("O", 2, 17),
        "edge_a": (10, 19),
        "edge_b": (28, 35),
    },
    {
        "pair_index": 9,
        "dataset": "EBL7",
        "sheet": "EBL7",
        "channel_a": 69,
        "channel_b": 86,
        "source_a": ("H", 2, 27),
        "source_b": ("I", 2, 25),
        "edge_a": (2, 14),
        "edge_b": (15, 26),
    },
    {
        "pair_index": 10,
        "dataset": "EBL7",
        "sheet": "EBL7",
        "channel_a": 100,
        "channel_b": 120,
        "source_a": ("M", 2, 15),
        "source_b": ("N", 2, 11),
        "edge_a": (27, 33),
        "edge_b": (34, 38),
    },
]

def save_figure(fig: plt.Figure, stem: Path, exact_size: bool = False) -> None:
    for suffix in (".svg", ".pdf", ".png"):
        kwargs = {"dpi": 600} if suffix == ".png" else {}
        if not exact_size:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)


def integer_set(ws: openpyxl.worksheet.worksheet.Worksheet, spec: tuple[str, int, int]) -> set[int]:
    column, start, stop = spec
    values = set()
    for row in range(start, stop + 1):
        value = ws[f"{column}{row}"].value
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.add(int(value))
    return values


def edge_set(ws: openpyxl.worksheet.worksheet.Worksheet, rows: tuple[int, int]) -> set[tuple[int, int]]:
    values = set()
    for row in range(rows[0], rows[1] + 1):
        source = ws[f"C{row}"].value
        target = ws[f"D{row}"].value
        if isinstance(source, (int, float)) and isinstance(target, (int, float)):
            values.add((int(source), int(target)))
    return values


def workbook_summary(wb: openpyxl.Workbook) -> pd.DataFrame:
    ws = wb["10pair_connectivity"]
    rows = []
    for row in range(2, ws.max_row + 1, 2):
        if ws[f"A{row}"].value is None:
            continue
        rows.append(
            {
                "pair_index": len(rows) + 1,
                "workbook_common_edges": int(ws[f"C{row}"].value or 0),
                "workbook_channel_specific_edges": int(ws[f"D{row}"].value or 0),
                "workbook_common_source_units": int(ws[f"E{row}"].value or 0),
                "workbook_channel_specific_source_units": int(ws[f"F{row}"].value or 0),
            }
        )
    return pd.DataFrame(rows)


def category_rows(
    config: dict,
    entity_type: str,
    values_a: set,
    values_b: set,
) -> list[dict]:
    categories = {
        "common": values_a & values_b,
        "channel_a_only": values_a - values_b,
        "channel_b_only": values_b - values_a,
    }
    rows = []
    for category, values in categories.items():
        for value in sorted(values):
            if entity_type == "connectivity_edge":
                identity = f"{value[0]}->{value[1]}"
                source_unit, target_unit = value
            else:
                identity = str(value)
                source_unit, target_unit = value, None
            rows.append(
                {
                    "pair_index": config["pair_index"],
                    "dataset": config["dataset"],
                    "channel_a": config["channel_a"],
                    "channel_b": config["channel_b"],
                    "entity_type": entity_type,
                    "category": category,
                    "identity": identity,
                    "source_unit": source_unit,
                    "target_unit": target_unit,
                }
            )
    return rows

def main():
    data_dir = pub_dir = OUT
    wb = openpyxl.load_workbook(S11_WORKBOOK, data_only=True, read_only=False)
    summary_target = workbook_summary(wb)
    membership_rows = []
    count_rows = []

    for config in PAIR_CONFIGS:
        ws = wb[config["sheet"]]
        source_a = integer_set(ws, config["source_a"])
        source_b = integer_set(ws, config["source_b"])
        edges_a = edge_set(ws, config["edge_a"])
        edges_b = edge_set(ws, config["edge_b"])
        membership_rows.extend(category_rows(config, "connectivity_source_unit", source_a, source_b))
        membership_rows.extend(category_rows(config, "connectivity_edge", edges_a, edges_b))
        count_rows.append(
            {
                "pair_index": config["pair_index"],
                "dataset": config["dataset"],
                "channel_a": config["channel_a"],
                "channel_b": config["channel_b"],
                "derived_common_source_units": len(source_a & source_b),
                "derived_channel_a_only_source_units": len(source_a - source_b),
                "derived_channel_b_only_source_units": len(source_b - source_a),
                "derived_common_edges": len(edges_a & edges_b),
                "derived_channel_a_only_edges": len(edges_a - edges_b),
                "derived_channel_b_only_edges": len(edges_b - edges_a),
            }
        )

    membership = pd.DataFrame(membership_rows)
    counts = pd.DataFrame(count_rows).merge(summary_target, on="pair_index", how="left")
    counts["derived_channel_specific_source_units"] = (
        counts["derived_channel_a_only_source_units"] + counts["derived_channel_b_only_source_units"]
    )
    counts["derived_channel_specific_edges"] = (
        counts["derived_channel_a_only_edges"] + counts["derived_channel_b_only_edges"]
    )
    counts["source_unit_summary_match"] = (
        (counts["derived_common_source_units"] == counts["workbook_common_source_units"])
        & (
            counts["derived_channel_specific_source_units"]
            == counts["workbook_channel_specific_source_units"]
        )
    )
    counts["edge_summary_match"] = (
        (counts["derived_common_edges"] == counts["workbook_common_edges"])
        & (counts["derived_channel_specific_edges"] == counts["workbook_channel_specific_edges"])
    )
    membership.to_csv(data_dir / "SF09_detailed_workbook_set_membership.csv", index=False)
    counts.to_csv(data_dir / "SF09_detailed_vs_summary_count_audit.csv", index=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    x = counts["pair_index"].to_numpy(dtype=float)
    panel_specs = [
        (
            "derived_common_source_units",
            "derived_channel_a_only_source_units",
            "derived_channel_b_only_source_units",
            "Source units",
        ),
        (
            "derived_common_edges",
            "derived_channel_a_only_edges",
            "derived_channel_b_only_edges",
            "Connectivity edges",
        ),
    ]

    def draw_original_bar(
        ax: plt.Axes,
        common_col: str,
        a_col: str,
        b_col: str,
        ylabel: str,
    ) -> None:
        common = counts[common_col].to_numpy(dtype=float)
        a_only = counts[a_col].to_numpy(dtype=float)
        b_only = counts[b_col].to_numpy(dtype=float)
        union = common + a_only + b_only
        channel_a_total = common + a_only
        channel_b_total = common + b_only
        ax.bar(
            x,
            union,
            width=0.68,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            zorder=1,
        )
        ax.bar(
            x,
            common,
            width=0.68,
            facecolor="0.82",
            edgecolor="black",
            linewidth=0.45,
            zorder=2,
        )
        for xi, shared, a_count, b_count, color in zip(x, common, a_only, b_only, colors):
            divider = shared + a_count
            ax.hlines(
                divider,
                xi - 0.34,
                xi + 0.34,
                colors="black",
                linewidth=0.55,
                zorder=3,
            )
            ax.scatter(
                xi,
                shared + a_count / 2,
                marker="^",
                s=17,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                zorder=4,
                clip_on=False,
            )
            ax.scatter(
                xi,
                divider + b_count / 2,
                marker="*",
                s=22,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                zorder=4,
                clip_on=False,
            )
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_xticks(x)
        ax.tick_params(width=0.6, length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)

    legend_handles = [
        Patch(facecolor="0.82", edgecolor="black", label="Common"),
        Patch(facecolor="white", edgecolor="black", label="Union"),
        Line2D([], [], marker="^", linestyle="none", color="black", markersize=4, label="Channel A only"),
        Line2D([], [], marker="*", linestyle="none", color="black", markersize=5, label="Channel B only"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(3.2, 5.0), sharex=True, constrained_layout=False)
    for ax, spec in zip(axes, panel_specs):
        draw_original_bar(ax, *spec)
    axes[0].tick_params(labelbottom=False)
    axes[0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=5,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.9,
        handlelength=1.4,
    )
    axes[1].set_xlabel("Stimulation-channel pair")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.10, top=0.94, hspace=0.34)
    save_figure(fig, pub_dir / "SF09bc_original_style_channel_markers_3p2x5in", exact_size=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    OUT = parser.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    main()
