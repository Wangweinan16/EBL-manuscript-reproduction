#!/usr/bin/env python3
"""
COMSOL two-trace capacitance exports → LTspice-compatible master CSV
====================================================================

**What this script does**

1. **Inter-trace coupling (Cx)** — reads ``COMSOL_Crosstalk_coupling_only.txt``,
   a COMSOL table with **one scalar per geometry** (1×1 inverse Maxwell matrix
   for the coupling-only physics). That value is stored as ``C_mut_pf_per_m``
   (pF/m), matching the ANSYS “coupling-only” column.

2. **Shunt / self capacitances (C_g1, C_g2)** — reads
   ``COMSOL_Crosstalk_coupling_sim.txt``, a **2×2** inverse Maxwell export: one
   data row per geometry plus a ``NaN`` / ``NaN`` symmetry row. The diagonal
   terms ``C11`` and ``C22`` (F/m) become ``C_g1_pf_per_m`` and ``C_g2_pf_per_m``.

3. **Merge** — joins the two tables on ``(width_nm, gap_ratio, frequency_hz)``
   using the same rules as ``ltspice_xtalk_postprocess.merge_coupling_and_shunt_master_dataframes``:
   final ``C_mut`` comes from the coupling-only file; ``C_g1``/``C_g2`` from the
   shunt (2×2) file; all conductances are **0** (electrostatic COMSOL).

4. **Electrostatic → AC frequency tag** — COMSOL exports have no frequency.
   Every row is tagged with ``--electrostatic-freq-hz`` (default **1000**) so
   downstream LTspice batch AC analysis has a defined point (same convention as
   the ANSYS master CSV).

5. **Outputs** — writes ``comsol_two_trace_master.csv`` under ``--outdir``,
   coupling vs LW plots, **shunt** plots (``C_g1`` / ``C_g2`` vs LW in two
   stacked panels), and prints absolute paths.

**Where to set paths**

* **Default layout** (recommended): put both COMSOL ``.txt`` files under
  ``COMSOL_output/`` next to this script, with the default names above. Then::

    python comsol_crosstalk_postprocess.py

* **Custom files or folder**::

    python comsol_crosstalk_postprocess.py ^
      --coupling-txt path/to/coupling_only.txt ^
      --shunt-txt path/to/coupling_sim.txt ^
      --outdir path/to/COMSOL_output

* **Coupling-only CSV** (no shunt merge): use ``--coupling-only`` with a single
  1×1 or legacy 2×2-only table (see ``--input-txt``).

**Downstream LTspice**

Run the batch pipeline on the combined master (example)::

    python ltspice_xtalk_postprocess.py ^
      --simulate-master-csv COMSOL_output/comsol_two_trace_master.csv ^
      --outdir ltspice_results_from_comsol

See ``COMSOL_output/README.md`` for a short end-to-end checklist.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from ltspice_xtalk_postprocess import merge_coupling_and_shunt_master_dataframes

# Default paths: data lives under COMSOL_output/ (see module docstring).
DEFAULT_COMSOL_DIR = pathlib.Path("COMSOL_output")
DEFAULT_COUPLING_TXT = DEFAULT_COMSOL_DIR / "COMSOL_Crosstalk_coupling_only.txt"
DEFAULT_SHUNT_TXT = DEFAULT_COMSOL_DIR / "COMSOL_Crosstalk_coupling_sim.txt"
DEFAULT_COMSOL_V1_DIR = pathlib.Path(r"D:\Simulation\Xtalk\xtalk_LTSpice_scripts\COMSOL_output1")
DEFAULT_COMSOL_V1_COUPLING_TXT = DEFAULT_COMSOL_V1_DIR / "XT_infinite_pbs_1terminal.txt"
DEFAULT_COMSOL_V1_SHUNT_TXT = DEFAULT_COMSOL_V1_DIR / "XT_infinite_pbs_outgnd.txt"
DEFAULT_COMSOL_V1_G_TXT = DEFAULT_COMSOL_V1_DIR / "XT_infinite_pbs_G_2t.txt"
DEFAULT_COMSOL_TEMP_DIR = pathlib.Path(r"D:\Simulation\Xtalk\xtalk_LTSpice_scripts\COMSOL_output_temp")
DEFAULT_TEMP_COUPLING_TXT = DEFAULT_COMSOL_DIR / "COMSOL_Crosstalk_coupling_only.txt"
DEFAULT_TEMP_SHUNT_C_TXT = DEFAULT_COMSOL_V1_DIR / "XT_infinite_pbs_outgnd.txt"
DEFAULT_TEMP_SHUNT_G_TXT = DEFAULT_COMSOL_V1_DIR / "XT_infinite_pbs_G_2t.txt"
MASTER_CSV_NAME = "comsol_two_trace_master.csv"
COUPLING_ONLY_MASTER_NAME = "comsol_coupling_only_master.csv"


def format_gap_ratio_token(gap_ratio: float) -> str:
    text = f"{float(gap_ratio):g}".replace(".", "p")
    return text.replace("-", "m")


def format_gap_ratio_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.12g")


def write_generation_note(outdir: pathlib.Path, note_text: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "GENERATION_NOTE.txt").write_text(note_text.strip() + "\n", encoding="utf-8")


def make_case_name(width_nm: float, gap_ratio: float) -> str:
    gap_token = format_gap_ratio_token(gap_ratio)
    return f"w{int(round(width_nm))}nm_r{gap_token}"


def _split_header_line(line: str) -> List[str]:
    rest = line[1:].strip() if line.lstrip().startswith("%") else line.strip()
    parts = [p.strip() for p in re.split(r"\s{2,}", rest) if p.strip()]
    return parts


def _read_comsol_text_table(path: pathlib.Path) -> Tuple[str, List[str], List[List[str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    model_note = ""
    header_parts: List[str] | None = None
    numeric_rows: List[List[str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            if "Model:" in line or "model:" in line:
                model_note = line.lstrip("%").strip()
            lower = line.lower()
            if "trace_w" in lower and "gap_ratio" in lower:
                header_parts = _split_header_line(line)
            continue
        tokens = line.split()
        if len(tokens) == 2:
            # COMSOL 2x2 inverse-Maxwell tables are often printed as two physical lines per
            # geometry: the second line omits the first two "key" columns (trace_w, gap_ratio).
            # Example:
            #   25  0.5  C11  C12
            #           C21  C22
            # Normalize those continuation rows to a 4-column row with NaN keys so downstream
            # parsing can pair them.
            numeric_rows.append(["nan", "nan", tokens[0], tokens[1]])
            continue
        if len(tokens) < 3:
            continue
        numeric_rows.append(tokens)

    if header_parts is None:
        raise ValueError(
            f"Could not locate a COMSOL header row containing trace_w and gap_ratio in {path}"
        )
    return model_note, header_parts, numeric_rows


def parse_comsol_1x1_coupling_only(
    model_note: str, header_parts: List[str], numeric_rows: List[List[str]]
) -> pd.DataFrame:
    widths = {len(r) for r in numeric_rows}
    if widths != {3}:
        raise ValueError(
            f"Expected 3 columns per row for 1x1 COMSOL export in coupling-only file, got {sorted(widths)}"
        )

    raw_df = pd.DataFrame(
        numeric_rows,
        columns=["trace_w_token", "gap_ratio_token", "C_coupling_F_per_m"],
    )
    for col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    primary = raw_df.dropna(subset=["trace_w_token", "gap_ratio_token"]).copy()
    primary = primary.rename(columns={"trace_w_token": "width_nm", "gap_ratio_token": "gap_ratio"})
    primary["gap_nm"] = primary["width_nm"] * primary["gap_ratio"]
    c_mut = primary["C_coupling_F_per_m"].astype(float)
    primary["C_mut_pf_per_m"] = c_mut * 1.0e12
    primary["matrix_mode"] = "comsol_coupling_only_1x1"
    primary["comsol_table_columns"] = " | ".join(header_parts)
    primary["model_header_note"] = model_note
    return primary.sort_values(["gap_ratio", "width_nm"]).reset_index(drop=True)


def parse_comsol_2x2_inverse_maxwell(
    model_note: str, header_parts: List[str], numeric_rows: List[List[str]]
) -> pd.DataFrame:
    widths = {len(r) for r in numeric_rows}
    if widths != {4}:
        raise ValueError(
            f"Expected 4 columns per row for 2x2 COMSOL export in shunt file, got {sorted(widths)}"
        )

    raw_df = pd.DataFrame(
        numeric_rows,
        columns=["trace_w_token", "gap_ratio_token", "c_low_token", "c_high_token"],
    )

    for col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    primary = raw_df.dropna(subset=["trace_w_token", "gap_ratio_token"]).copy()
    primary = primary.rename(
        columns={
            "trace_w_token": "width_nm",
            "gap_ratio_token": "gap_ratio",
            "c_low_token": "C_col0_F_per_m",
            "c_high_token": "C_col1_F_per_m",
        }
    )

    companions: dict[int, Tuple[float, float]] = {}
    for idx in primary.index:
        next_pos = int(idx) + 1
        if next_pos >= len(raw_df):
            continue
        row_next = raw_df.iloc[next_pos]
        if pd.isna(row_next["trace_w_token"]) and pd.isna(row_next["gap_ratio_token"]):
            companions[int(idx)] = (
                float(row_next["c_low_token"]),
                float(row_next["c_high_token"]),
            )

    c22_list: List[float] = []
    c21_check: List[float] = []
    for idx in primary.index:
        pair = companions.get(idx)
        if pair is None:
            c22_list.append(float("nan"))
            c21_check.append(float("nan"))
            continue
        c21, c22 = pair
        c22_list.append(c22)
        c21_check.append(c21)

    primary["C11_F_per_m"] = primary["C_col0_F_per_m"].astype(float)
    primary["C12_F_per_m"] = primary["C_col1_F_per_m"].astype(float)
    primary["C21_F_per_m"] = pd.Series(c21_check, index=primary.index)
    primary["C22_F_per_m"] = pd.Series(c22_list, index=primary.index)

    mismatch = primary.dropna(subset=["C12_F_per_m", "C21_F_per_m"]).copy()
    if not mismatch.empty:
        rel = (
            (mismatch["C12_F_per_m"] - mismatch["C21_F_per_m"]).abs()
            / mismatch["C12_F_per_m"].abs().clip(lower=1e-30)
        )
        if (rel > 1e-6).any():
            bad = mismatch.loc[rel > 1e-6].head(3)
            raise ValueError(
                "COMSOL matrix symmetry check failed between C12 and C21 on rows:\n"
                f"{bad[['width_nm', 'gap_ratio', 'C12_F_per_m', 'C21_F_per_m']]}"
            )

    primary["gap_nm"] = primary["width_nm"] * primary["gap_ratio"]
    primary["C_mut_pf_per_m"] = primary["C12_F_per_m"] * 1.0e12
    primary["C_g1_pf_per_m"] = primary["C11_F_per_m"] * 1.0e12
    primary["C_g2_pf_per_m"] = primary["C22_F_per_m"] * 1.0e12
    primary["matrix_mode"] = "comsol_inverse_maxwell_2x2_table"
    primary["comsol_table_columns"] = " | ".join(header_parts)
    primary["model_header_note"] = model_note
    return primary.sort_values(["gap_ratio", "width_nm"]).reset_index(drop=True)


def parse_comsol_table_export(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    model_note, header_parts, numeric_rows = _read_comsol_text_table(path)
    if not numeric_rows:
        raise ValueError(f"No numeric data rows found in {path}")

    widths = {len(r) for r in numeric_rows}
    if widths == {3}:
        return parse_comsol_1x1_coupling_only(model_note, header_parts, numeric_rows)
    if widths == {4}:
        return parse_comsol_2x2_inverse_maxwell(model_note, header_parts, numeric_rows)

    raise ValueError(
        f"Unsupported COMSOL table in {path}: expected 3 columns (1x1) or 4 columns (2x2 + symmetry), got {sorted(widths)}"
    )


def coupling_parsed_to_merge_dataframe(parsed: pd.DataFrame, source_path: pathlib.Path, freq_hz: float) -> pd.DataFrame:
    rows = []
    for rec in parsed.to_dict(orient="records"):
        width_nm = float(rec["width_nm"])
        gap_ratio = float(rec["gap_ratio"])
        case_name = make_case_name(width_nm, gap_ratio)
        rows.append(
            {
                "case_name": case_name,
                "design_name": f"COMSOL_coupling_{case_name}",
                "width_nm": width_nm,
                "gap_ratio": gap_ratio,
                "gap_nm": float(rec["gap_nm"]),
                "frequency_hz": float(freq_hz),
                "source_name": str(rec.get("model_header_note", "")),
                "source_file": source_path.name,
                "matrix_mode": str(rec["matrix_mode"]),
                "C_mut_pf_per_m": float(rec["C_mut_pf_per_m"]),
                "C_g1_pf_per_m": 0.0,
                "C_g2_pf_per_m": 0.0,
                "G_mut_mS_per_m": 0.0,
                "G_g1_mS_per_m": 0.0,
                "G_g2_mS_per_m": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency_hz", "gap_ratio", "width_nm"]).reset_index(drop=True)


def shunt_parsed_to_merge_dataframe(parsed: pd.DataFrame, source_path: pathlib.Path, freq_hz: float) -> pd.DataFrame:
    rows = []
    for rec in parsed.to_dict(orient="records"):
        width_nm = float(rec["width_nm"])
        gap_ratio = float(rec["gap_ratio"])
        case_name = make_case_name(width_nm, gap_ratio)
        rows.append(
            {
                "case_name": case_name,
                "design_name": f"COMSOL_shunt_{case_name}",
                "width_nm": width_nm,
                "gap_ratio": gap_ratio,
                "gap_nm": float(rec["gap_nm"]),
                "frequency_hz": float(freq_hz),
                "source_name": str(rec.get("model_header_note", "")),
                "source_file": source_path.name,
                "matrix_mode": str(rec["matrix_mode"]),
                "shunt_mapping": "comsol_2x2_diagonals",
                "C_mut_pf_per_m": float(rec["C_mut_pf_per_m"]),
                "C_g1_pf_per_m": float(rec["C_g1_pf_per_m"]),
                "C_g2_pf_per_m": float(rec["C_g2_pf_per_m"]),
                "G_mut_mS_per_m": 0.0,
                "G_g1_mS_per_m": 0.0,
                "G_g2_mS_per_m": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency_hz", "gap_ratio", "width_nm"]).reset_index(drop=True)


def build_single_file_master_dataframe(parsed: pd.DataFrame, source_path: pathlib.Path) -> pd.DataFrame:
    """Legacy single-table master (1x1 or 2x2-only) for diagnostics."""
    rows = []
    for rec in parsed.to_dict(orient="records"):
        width_nm = float(rec["width_nm"])
        gap_ratio = float(rec["gap_ratio"])
        case_name = make_case_name(width_nm, gap_ratio)
        mm = str(rec.get("matrix_mode", ""))
        if mm == "comsol_coupling_only_1x1":
            c_mut = float(rec["C_mut_pf_per_m"])
            cg1 = float("nan")
            cg2 = float("nan")
            c11 = c12 = c22 = float("nan")
        else:
            c_mut = float(rec["C_mut_pf_per_m"])
            cg1 = float(rec["C_g1_pf_per_m"])
            cg2 = float(rec["C_g2_pf_per_m"])
            c11 = rec.get("C11_F_per_m")
            c12 = rec.get("C12_F_per_m")
            c22 = rec.get("C22_F_per_m")
        rows.append(
            {
                "case_name": case_name,
                "design_name": f"COMSOL_2T_{case_name}",
                "width_nm": width_nm,
                "gap_ratio": gap_ratio,
                "gap_nm": float(rec["gap_nm"]),
                "frequency_hz": float("nan"),
                "source_name": str(rec.get("model_header_note", "")),
                "source_file": source_path.name,
                "n_conductors": 1,
                "conductors": "COMSOL_coupling_only_1x1" if "1x1" in mm else "COMSOL_two_trace_table",
                "matrix_mode": mm,
                "C_mut_pf_per_m": c_mut,
                "C_g1_pf_per_m": cg1,
                "C_g2_pf_per_m": cg2,
                "G_mut_mS_per_m": 0.0,
                "G_g1_mS_per_m": 0.0,
                "G_g2_mS_per_m": 0.0,
                "C11_F_per_m": c11,
                "C12_F_per_m": c12,
                "C22_F_per_m": c22,
                "comsol_table_columns": rec.get("comsol_table_columns", ""),
            }
        )
    column_order = [
        "case_name",
        "design_name",
        "width_nm",
        "gap_ratio",
        "gap_nm",
        "frequency_hz",
        "source_name",
        "source_file",
        "n_conductors",
        "conductors",
        "matrix_mode",
        "C_mut_pf_per_m",
        "C_g1_pf_per_m",
        "C_g2_pf_per_m",
        "G_mut_mS_per_m",
        "G_g1_mS_per_m",
        "G_g2_mS_per_m",
        "C11_F_per_m",
        "C12_F_per_m",
        "C22_F_per_m",
        "comsol_table_columns",
    ]
    return pd.DataFrame(rows)[column_order]


def plot_coupling_vs_lw(master_df: pd.DataFrame, out_path: pathlib.Path, y_log_scale: bool = False) -> None:
    plot_df = master_df.sort_values(["gap_ratio", "width_nm"]).copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)

    for gap_ratio, curve_df in plot_df.groupby("gap_ratio", sort=True):
        curve_df = curve_df.sort_values("width_nm")
        if y_log_scale:
            # COMSOL inverse-Maxwell off-diagonal terms (C12) are typically negative.
            # For log-y visualization, plot -C_mut so values are positive and comparable on a log axis.
            y_values = (-curve_df["C_mut_pf_per_m"]).clip(lower=1e-12)
        else:
            y_values = curve_df["C_mut_pf_per_m"]
        ax.semilogx(
            curve_df["width_nm"],
            y_values,
            marker="o",
            linewidth=1.8,
            label=f"G/LW = {format_gap_ratio_label(float(gap_ratio))}",
        )

    ax.set_xlabel("LW (nm)")
    ax.set_ylabel("-C_mut (pF/m)" if y_log_scale else "Inter-trace C_mut (pF/m)")
    if y_log_scale:
        ax.set_yscale("log")

    n_lines = int(plot_df["gap_ratio"].nunique())
    suffix = " [log y]" if y_log_scale else ""
    ax.set_title(f"COMSOL combined master: {'-C_mut' if y_log_scale else 'C_mut'} vs LW ({n_lines} G/LW curves){suffix}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Gap ratio")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def master_has_shunt_caps(master_df: pd.DataFrame) -> bool:
    if master_df.empty:
        return False
    for col in ("C_g1_pf_per_m", "C_g2_pf_per_m"):
        if col not in master_df.columns:
            return False
        if not master_df[col].notna().any():
            return False
    return True


def plot_shunt_vs_lw(master_df: pd.DataFrame, out_path: pathlib.Path, y_log_scale: bool = False) -> None:
    """Shunt capacitances from the 2×2 COMSOL diagonal terms (merged master columns C_g1, C_g2)."""
    plot_df = master_df.sort_values(["gap_ratio", "width_nm"]).copy()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.8, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    n_lines = int(plot_df["gap_ratio"].nunique())
    suffix = " [log y]" if y_log_scale else ""

    for gap_ratio, curve_df in plot_df.groupby("gap_ratio", sort=True):
        curve_df = curve_df.sort_values("width_nm")
        y1 = curve_df["C_g1_pf_per_m"].clip(lower=1e-12) if y_log_scale else curve_df["C_g1_pf_per_m"]
        y2 = curve_df["C_g2_pf_per_m"].clip(lower=1e-12) if y_log_scale else curve_df["C_g2_pf_per_m"]
        label = f"G/LW = {format_gap_ratio_label(float(gap_ratio))}"
        ax1.semilogx(curve_df["width_nm"], y1, marker="o", linewidth=1.8, label=label)
        ax2.semilogx(curve_df["width_nm"], y2, marker="o", linewidth=1.8, label=label)

    ax1.set_ylabel("C_g1 (pF/m)")
    ax1.set_title(f"COMSOL shunt: diagonal C11 vs LW ({n_lines} G/LW curves){suffix}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(title="Gap ratio", loc="best", fontsize=8)

    ax2.set_xlabel("LW (nm)")
    ax2.set_ylabel("C_g2 (pF/m)")
    ax2.set_title(f"COMSOL shunt: diagonal C22 vs LW ({n_lines} G/LW curves){suffix}")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(title="Gap ratio", loc="best", fontsize=8)

    if y_log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gmut_vs_lw(master_df: pd.DataFrame, out_path: pathlib.Path, y_log_scale: bool = False) -> None:
    plot_df = master_df.sort_values(["gap_ratio", "width_nm"]).copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)

    for gap_ratio, curve_df in plot_df.groupby("gap_ratio", sort=True):
        curve_df = curve_df.sort_values("width_nm")
        if y_log_scale:
            y_values = (-curve_df["G_mut_mS_per_m"]).abs().clip(lower=1e-30)
        else:
            y_values = curve_df["G_mut_mS_per_m"]
        ax.semilogx(
            curve_df["width_nm"],
            y_values,
            marker="o",
            linewidth=1.8,
            label=f"G/LW = {format_gap_ratio_label(float(gap_ratio))}",
        )

    ax.set_xlabel("LW (nm)")
    ax.set_ylabel("-G_mut (mS/m)" if y_log_scale else "G_mut (mS/m)")
    if y_log_scale:
        ax.set_yscale("log")

    n_lines = int(plot_df["gap_ratio"].nunique())
    suffix = " [log y]" if y_log_scale else ""
    ax.set_title(f"COMSOL combined master: {'-G_mut' if y_log_scale else 'G_mut'} vs LW ({n_lines} G/LW curves){suffix}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Gap ratio")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_shunt_g_vs_lw(master_df: pd.DataFrame, out_path: pathlib.Path, y_log_scale: bool = False) -> None:
    plot_df = master_df.sort_values(["gap_ratio", "width_nm"]).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, constrained_layout=True)
    n_lines = int(plot_df["gap_ratio"].nunique())
    suffix = " [log y]" if y_log_scale else ""

    for gap_ratio, curve_df in plot_df.groupby("gap_ratio", sort=True):
        curve_df = curve_df.sort_values("width_nm")
        y1 = curve_df["G_g1_mS_per_m"].abs().clip(lower=1e-30) if y_log_scale else curve_df["G_g1_mS_per_m"]
        y2 = curve_df["G_g2_mS_per_m"].abs().clip(lower=1e-30) if y_log_scale else curve_df["G_g2_mS_per_m"]
        label = f"G/LW = {format_gap_ratio_label(float(gap_ratio))}"
        ax1.semilogx(curve_df["width_nm"], y1, marker="o", linewidth=1.8, label=label)
        ax2.semilogx(curve_df["width_nm"], y2, marker="o", linewidth=1.8, label=label)

    ax1.set_ylabel("G_g1 (mS/m)")
    ax1.set_title(f"COMSOL shunt: diagonal G11 vs LW ({n_lines} G/LW curves){suffix}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(title="Gap ratio", loc="best", fontsize=8)

    ax2.set_xlabel("LW (nm)")
    ax2.set_ylabel("G_g2 (mS/m)")
    ax2.set_title(f"COMSOL shunt: diagonal G22 vs LW ({n_lines} G/LW curves){suffix}")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(title="Gap ratio", loc="best", fontsize=8)

    if y_log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_combined_mode(
    coupling_txt: pathlib.Path,
    shunt_txt: pathlib.Path,
    outdir: pathlib.Path,
    electrostatic_freq_hz: float,
) -> pd.DataFrame:
    c_note, c_hdr, c_rows = _read_comsol_text_table(coupling_txt)
    s_note, s_hdr, s_rows = _read_comsol_text_table(shunt_txt)
    parsed_coupling = parse_comsol_1x1_coupling_only(c_note, c_hdr, c_rows)
    parsed_shunt = parse_comsol_2x2_inverse_maxwell(s_note, s_hdr, s_rows)

    coupling_df = coupling_parsed_to_merge_dataframe(parsed_coupling, coupling_txt.resolve(), electrostatic_freq_hz)
    shunt_df = shunt_parsed_to_merge_dataframe(parsed_shunt, shunt_txt.resolve(), electrostatic_freq_hz)

    master_df = merge_coupling_and_shunt_master_dataframes(coupling_df, shunt_df)
    master_df["model_variant"] = "comsol_coupling_only_1x1_plus_comsol_2x2_shunt"
    return master_df


def _merge_in_g_columns(
    base_df: pd.DataFrame,
    g_df: pd.DataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    keys = ["width_nm", "gap_ratio", "frequency_hz"]
    needed = list({*column_map.values()})
    merged = base_df.merge(
        g_df[keys + needed],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_g"),
    )
    for base_col, g_col in column_map.items():
        # If base_df already has the column (often initialized to 0),
        # pandas will suffix the right-hand value with "_g".
        rhs_col = f"{g_col}_g" if f"{g_col}_g" in merged.columns else g_col
        merged[base_col] = merged[rhs_col]
    drop_cols = [c for c in needed if c not in column_map.keys() and c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def parse_comsol_2x2_inverse_matrix_as_conductance(
    model_note: str, header_parts: List[str], numeric_rows: List[List[str]]
) -> pd.DataFrame:
    parsed = parse_comsol_2x2_inverse_maxwell(model_note, header_parts, numeric_rows)
    df = parsed[["width_nm", "gap_ratio", "gap_nm", "C11_F_per_m", "C12_F_per_m", "C22_F_per_m"]].copy()
    df["G_mut_mS_per_m"] = df["C12_F_per_m"].astype(float) * 1.0e3
    df["G_g1_mS_per_m"] = df["C11_F_per_m"].astype(float) * 1.0e3
    df["G_g2_mS_per_m"] = df["C22_F_per_m"].astype(float) * 1.0e3
    df["matrix_mode"] = "comsol_inverse_matrix_2x2_table_conductance"
    df["comsol_table_columns"] = " | ".join(header_parts)
    df["model_header_note"] = model_note
    return df.sort_values(["gap_ratio", "width_nm"]).reset_index(drop=True)


def build_g_2x2_dataframe_for_merge(parsed_g: pd.DataFrame, source_path: pathlib.Path, freq_hz: float) -> pd.DataFrame:
    rows = []
    for rec in parsed_g.to_dict(orient="records"):
        width_nm = float(rec["width_nm"])
        gap_ratio = float(rec["gap_ratio"])
        case_name = make_case_name(width_nm, gap_ratio)
        rows.append(
            {
                "case_name": case_name,
                "design_name": f"COMSOL_G_{case_name}",
                "width_nm": width_nm,
                "gap_ratio": gap_ratio,
                "gap_nm": float(rec["gap_nm"]),
                "frequency_hz": float(freq_hz),
                "source_name": str(rec.get("model_header_note", "")),
                "source_file": source_path.name,
                "matrix_mode": str(rec.get("matrix_mode", "comsol_inverse_matrix_2x2_table_conductance")),
                "G_mut_mS_per_m": float(rec["G_mut_mS_per_m"]),
                "G_g1_mS_per_m": float(rec["G_g1_mS_per_m"]),
                "G_g2_mS_per_m": float(rec["G_g2_mS_per_m"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency_hz", "gap_ratio", "width_nm"]).reset_index(drop=True)


def run_v1_three_file_mode(
    coupling_cmut_txt: pathlib.Path,
    shunt_cg_txt: pathlib.Path,
    g_2t_txt: pathlib.Path,
    electrostatic_freq_hz: float,
) -> pd.DataFrame:
    c_note, c_hdr, c_rows = _read_comsol_text_table(coupling_cmut_txt)
    parsed_cmut = parse_comsol_1x1_coupling_only(c_note, c_hdr, c_rows)
    coupling_df = coupling_parsed_to_merge_dataframe(parsed_cmut, coupling_cmut_txt.resolve(), electrostatic_freq_hz)

    s_note, s_hdr, s_rows = _read_comsol_text_table(shunt_cg_txt)
    parsed_cg = parse_comsol_2x2_inverse_maxwell(s_note, s_hdr, s_rows)
    shunt_df = shunt_parsed_to_merge_dataframe(parsed_cg, shunt_cg_txt.resolve(), electrostatic_freq_hz)

    g_note, g_hdr, g_rows = _read_comsol_text_table(g_2t_txt)
    parsed_g = parse_comsol_2x2_inverse_matrix_as_conductance(g_note, g_hdr, g_rows)
    g_df = build_g_2x2_dataframe_for_merge(parsed_g, g_2t_txt.resolve(), electrostatic_freq_hz)

    coupling_df = _merge_in_g_columns(coupling_df, g_df, {"G_mut_mS_per_m": "G_mut_mS_per_m"})
    shunt_df = _merge_in_g_columns(
        shunt_df,
        g_df,
        {"G_g1_mS_per_m": "G_g1_mS_per_m", "G_g2_mS_per_m": "G_g2_mS_per_m"},
    )

    master_df = merge_coupling_and_shunt_master_dataframes(coupling_df, shunt_df)
    master_df["model_variant"] = "comsol_v1_cmut_plus_cg_plus_g_2t"
    master_df["source_file_cmut"] = coupling_cmut_txt.name
    master_df["source_file_cg"] = shunt_cg_txt.name
    master_df["source_file_g"] = g_2t_txt.name
    return master_df


def run_vtemp_mode(electrostatic_freq_hz: float) -> pd.DataFrame:
    """
    V_temp mixes sources across folders:
      - C_mut from COMSOL_output/COMSOL_Crosstalk_coupling_only.txt (1x1)
      - shunt C from COMSOL_output1/XT_infinite_pbs_outgnd.txt (2x2 diagonals)
      - shunt G from COMSOL_output1/XT_infinite_pbs_G_2t.txt (2x2 interpreted as S/m)
    """
    coupling_cmut_txt = DEFAULT_TEMP_COUPLING_TXT
    shunt_cg_txt = DEFAULT_TEMP_SHUNT_C_TXT
    g_2t_txt = DEFAULT_TEMP_SHUNT_G_TXT

    for p in (coupling_cmut_txt, shunt_cg_txt, g_2t_txt):
        if not p.exists():
            raise FileNotFoundError(p)

    c_note, c_hdr, c_rows = _read_comsol_text_table(coupling_cmut_txt)
    parsed_cmut = parse_comsol_1x1_coupling_only(c_note, c_hdr, c_rows)
    coupling_df = coupling_parsed_to_merge_dataframe(parsed_cmut, coupling_cmut_txt.resolve(), electrostatic_freq_hz)

    s_note, s_hdr, s_rows = _read_comsol_text_table(shunt_cg_txt)
    parsed_cg = parse_comsol_2x2_inverse_maxwell(s_note, s_hdr, s_rows)
    shunt_df = shunt_parsed_to_merge_dataframe(parsed_cg, shunt_cg_txt.resolve(), electrostatic_freq_hz)

    g_note, g_hdr, g_rows = _read_comsol_text_table(g_2t_txt)
    parsed_g = parse_comsol_2x2_inverse_matrix_as_conductance(g_note, g_hdr, g_rows)
    g_df = build_g_2x2_dataframe_for_merge(parsed_g, g_2t_txt.resolve(), electrostatic_freq_hz)

    # For V_temp: treat the 2-terminal G table as shunt G only.
    shunt_df = _merge_in_g_columns(
        shunt_df,
        g_df,
        {"G_g1_mS_per_m": "G_g1_mS_per_m", "G_g2_mS_per_m": "G_g2_mS_per_m"},
    )

    master_df = merge_coupling_and_shunt_master_dataframes(coupling_df, shunt_df)
    master_df["model_variant"] = "comsol_vtemp_cmut_from_v0_shunt_from_v1"
    master_df["source_file_cmut"] = coupling_cmut_txt.name
    master_df["source_file_cg"] = shunt_cg_txt.name
    master_df["source_file_g"] = g_2t_txt.name
    return master_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COMSOL coupling + shunt tables -> merged master CSV + plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring at the top of comsol_crosstalk_postprocess.py for full documentation.",
    )
    parser.add_argument(
        "--version",
        choices=("0", "1", "temp"),
        default=None,
        help="Select workflow: 0=V0 (two-file merge), 1=V1 (C_mut + C_g + G from three files under COMSOL_output1), temp=V_temp mixed sources into COMSOL_output_temp. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--coupling-txt",
        type=pathlib.Path,
        default=DEFAULT_COUPLING_TXT,
        help=f"1x1 coupling-only COMSOL table. Default: {DEFAULT_COUPLING_TXT}",
    )
    parser.add_argument(
        "--shunt-txt",
        type=pathlib.Path,
        default=DEFAULT_SHUNT_TXT,
        help=f"2x2 shunt COMSOL table (diagonals -> C_g1/C_g2). Default: {DEFAULT_SHUNT_TXT}",
    )
    parser.add_argument(
        "--v1-coupling-txt",
        type=pathlib.Path,
        default=DEFAULT_COMSOL_V1_COUPLING_TXT,
        help=f"V1 input (C_mut only): COMSOL 1x1 mutual capacitance table. Default: {DEFAULT_COMSOL_V1_COUPLING_TXT}",
    )
    parser.add_argument(
        "--v1-shunt-txt",
        type=pathlib.Path,
        default=DEFAULT_COMSOL_V1_SHUNT_TXT,
        help=f"V1 input (C_g diagonals): COMSOL 2x2 inverse Maxwell table. Default: {DEFAULT_COMSOL_V1_SHUNT_TXT}",
    )
    parser.add_argument(
        "--v1-g-txt",
        type=pathlib.Path,
        default=DEFAULT_COMSOL_V1_G_TXT,
        help=f"V1 input (G matrix): COMSOL 2x2 inverse matrix table for conductance (S/m). Default: {DEFAULT_COMSOL_V1_G_TXT}",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=DEFAULT_COMSOL_DIR,
        help=f"Output directory for CSV and PNGs. Default: {DEFAULT_COMSOL_DIR}",
    )
    parser.add_argument(
        "--electrostatic-freq-hz",
        type=float,
        default=1000.0,
        help="Frequency (Hz) assigned to electrostatic COMSOL rows for downstream LTspice. Default: 1000.",
    )
    parser.add_argument(
        "--coupling-only",
        action="store_true",
        help="Only process --input-txt (single file: 1x1 or 4-column 2x2 export). Writes coupling-only master; no merge.",
    )
    parser.add_argument(
        "--input-txt",
        type=pathlib.Path,
        default=DEFAULT_COUPLING_TXT,
        help=f"Used with --coupling-only only. Default: {DEFAULT_COUPLING_TXT}",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    version = args.version
    if version is None:
        try:
            raw = input(
                "Select COMSOL postprocess version (0=V0 two-file merge, 1=V1 from COMSOL_output1, temp=V_temp -> COMSOL_output_temp): "
            ).strip()
        except EOFError:
            raw = ""
        version = raw if raw in ("0", "1", "temp") else "0"

    if version == "1":
        outdir = DEFAULT_COMSOL_V1_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        for p in (args.v1_coupling_txt, args.v1_shunt_txt, args.v1_g_txt):
            if not p.exists():
                print(f"ERROR: V1 input file not found: {p.resolve()}", file=sys.stderr)
                sys.exit(1)

        master_df = run_v1_three_file_mode(
            args.v1_coupling_txt,
            args.v1_shunt_txt,
            args.v1_g_txt,
            args.electrostatic_freq_hz,
        )
        master_path = outdir / MASTER_CSV_NAME
        write_csv(master_df, master_path)

        plot_path = outdir / "comsol_capacitive_coupling_vs_lw.png"
        plot_log_path = outdir / "comsol_capacitive_coupling_vs_lw_logy.png"
        plot_coupling_vs_lw(master_df, plot_path, y_log_scale=False)
        plot_coupling_vs_lw(master_df, plot_log_path, y_log_scale=True)

        shunt_path = outdir / "comsol_shunt_cap_vs_lw.png"
        shunt_log_path = outdir / "comsol_shunt_cap_vs_lw_logy.png"
        plot_shunt_vs_lw(master_df, shunt_path, y_log_scale=False)
        plot_shunt_vs_lw(master_df, shunt_log_path, y_log_scale=True)

        gmut_path = outdir / "comsol_conductive_coupling_gmut_vs_lw.png"
        gmut_log_path = outdir / "comsol_conductive_coupling_gmut_vs_lw_logy.png"
        plot_gmut_vs_lw(master_df, gmut_path, y_log_scale=False)
        plot_gmut_vs_lw(master_df, gmut_log_path, y_log_scale=True)

        gshunt_path = outdir / "comsol_shunt_g_vs_lw.png"
        gshunt_log_path = outdir / "comsol_shunt_g_vs_lw_logy.png"
        plot_shunt_g_vs_lw(master_df, gshunt_path, y_log_scale=False)
        plot_shunt_g_vs_lw(master_df, gshunt_log_path, y_log_scale=True)

        print(f"Saved V1 master CSV:      {master_path.resolve()}")
        print(f"Saved plot:               {plot_path.resolve()}")
        print(f"Saved log-y plot:         {plot_log_path.resolve()}")
        print(f"Saved shunt plot:         {shunt_path.resolve()}")
        print(f"Saved shunt log-y plot:   {shunt_log_path.resolve()}")
        print(f"Saved G_mut plot:         {gmut_path.resolve()}")
        print(f"Saved G_mut log-y plot:   {gmut_log_path.resolve()}")
        print(f"Saved shunt G plot:       {gshunt_path.resolve()}")
        print(f"Saved shunt G log-y plot: {gshunt_log_path.resolve()}")
        print(f"Rows: {len(master_df)}; electrostatic_freq_hz={args.electrostatic_freq_hz:g} Hz (tag for LTspice)")
        return

    if version == "temp":
        outdir = DEFAULT_COMSOL_TEMP_DIR
        outdir.mkdir(parents=True, exist_ok=True)

        master_df = run_vtemp_mode(args.electrostatic_freq_hz)
        master_path = outdir / MASTER_CSV_NAME
        write_csv(master_df, master_path)

        plot_path = outdir / "comsol_capacitive_coupling_vs_lw.png"
        plot_log_path = outdir / "comsol_capacitive_coupling_vs_lw_logy.png"
        plot_coupling_vs_lw(master_df, plot_path, y_log_scale=False)
        plot_coupling_vs_lw(master_df, plot_log_path, y_log_scale=True)

        shunt_path = outdir / "comsol_shunt_cap_vs_lw.png"
        shunt_log_path = outdir / "comsol_shunt_cap_vs_lw_logy.png"
        plot_shunt_vs_lw(master_df, shunt_path, y_log_scale=False)
        plot_shunt_vs_lw(master_df, shunt_log_path, y_log_scale=True)

        gshunt_path = outdir / "comsol_shunt_g_vs_lw.png"
        gshunt_log_path = outdir / "comsol_shunt_g_vs_lw_logy.png"
        plot_shunt_g_vs_lw(master_df, gshunt_path, y_log_scale=False)
        plot_shunt_g_vs_lw(master_df, gshunt_log_path, y_log_scale=True)

        note = f"""
V_temp generation note
======================

This master CSV and plots were generated by `comsol_crosstalk_postprocess.py` in version=temp mode.

Inputs:
- C_mut:   {DEFAULT_TEMP_COUPLING_TXT.resolve()}
- C_g:     {DEFAULT_TEMP_SHUNT_C_TXT.resolve()}  (2x2 diagonals -> C_g1, C_g2)
- G_shunt: {DEFAULT_TEMP_SHUNT_G_TXT.resolve()}  (2x2 inverse matrix interpreted as conductance; only diagonals used)

Key mapping:
- C_mut_pf_per_m comes from the COMSOL_output coupling-only 1x1 table.
- C_g1_pf_per_m, C_g2_pf_per_m come from the COMSOL_output1 outgnd 2x2 table.
- G_g1_mS_per_m, G_g2_mS_per_m come from the COMSOL_output1 G_2t table.
- G_mut_mS_per_m is left at 0 in this temporary mixed mode.

electrostatic_freq_hz tag: {args.electrostatic_freq_hz:g} Hz
"""
        write_generation_note(outdir, note)

        print(f"Saved V_temp master CSV:  {master_path.resolve()}")
        print(f"Saved plot:               {plot_path.resolve()}")
        print(f"Saved log-y plot:         {plot_log_path.resolve()}")
        print(f"Saved shunt plot:         {shunt_path.resolve()}")
        print(f"Saved shunt log-y plot:   {shunt_log_path.resolve()}")
        print(f"Saved shunt G plot:       {gshunt_path.resolve()}")
        print(f"Saved shunt G log-y plot: {gshunt_log_path.resolve()}")
        print(f"Saved note:               {(outdir / 'GENERATION_NOTE.txt').resolve()}")
        print(f"Rows: {len(master_df)}; electrostatic_freq_hz={args.electrostatic_freq_hz:g} Hz (tag for LTspice)")
        return

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.coupling_only:
        parsed = parse_comsol_table_export(args.input_txt)
        master_df = build_single_file_master_dataframe(parsed, args.input_txt.resolve())
        master_path = outdir / COUPLING_ONLY_MASTER_NAME
        write_csv(master_df, master_path)
        plot_coupling_vs_lw(master_df, outdir / "comsol_capacitive_coupling_vs_lw.png", y_log_scale=False)
        plot_coupling_vs_lw(master_df, outdir / "comsol_capacitive_coupling_vs_lw_logy.png", y_log_scale=True)
        if master_has_shunt_caps(master_df):
            plot_shunt_vs_lw(master_df, outdir / "comsol_shunt_cap_vs_lw.png", y_log_scale=False)
            plot_shunt_vs_lw(master_df, outdir / "comsol_shunt_cap_vs_lw_logy.png", y_log_scale=True)
            print(f"Saved shunt plot:          {(outdir / 'comsol_shunt_cap_vs_lw.png').resolve()}")
        print(f"Saved coupling-only master CSV: {master_path.resolve()}")
        return

    if not args.coupling_txt.exists():
        print(f"ERROR: coupling file not found: {args.coupling_txt.resolve()}", file=sys.stderr)
        sys.exit(1)
    if not args.shunt_txt.exists():
        print(f"ERROR: shunt file not found: {args.shunt_txt.resolve()}", file=sys.stderr)
        sys.exit(1)

    master_df = run_combined_mode(
        args.coupling_txt,
        args.shunt_txt,
        outdir,
        args.electrostatic_freq_hz,
    )

    master_path = outdir / MASTER_CSV_NAME
    write_csv(master_df, master_path)

    plot_path = outdir / "comsol_capacitive_coupling_vs_lw.png"
    plot_log_path = outdir / "comsol_capacitive_coupling_vs_lw_logy.png"
    plot_coupling_vs_lw(master_df, plot_path, y_log_scale=False)
    plot_coupling_vs_lw(master_df, plot_log_path, y_log_scale=True)

    shunt_path = outdir / "comsol_shunt_cap_vs_lw.png"
    shunt_log_path = outdir / "comsol_shunt_cap_vs_lw_logy.png"
    plot_shunt_vs_lw(master_df, shunt_path, y_log_scale=False)
    plot_shunt_vs_lw(master_df, shunt_log_path, y_log_scale=True)

    print(f"Saved merged master CSV: {master_path.resolve()}")
    print(f"Saved plot:              {plot_path.resolve()}")
    print(f"Saved log-y plot:        {plot_log_path.resolve()}")
    print(f"Saved shunt plot:        {shunt_path.resolve()}")
    print(f"Saved shunt log-y plot:  {shunt_log_path.resolve()}")
    print(f"Rows: {len(master_df)}; electrostatic_freq_hz={args.electrostatic_freq_hz:g} Hz (tag for LTspice)")


if __name__ == "__main__":
    main()
