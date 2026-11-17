"""
Build a standard COMSOL/LTspice master CSV from large-sweep COMSOL exports
at fixed PI conductivity 1e-9 S/m.
"""

from __future__ import annotations

import datetime as dt
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "COMSOL_output_large_sweep_1em9_inplane"
OUT_DIR = ROOT / "COMSOL_output_large_sweep_1em9_inplane_pipeline"
MASTER_CSV = OUT_DIR / "comsol_two_trace_master.csv"
DEFAULT_PLANE_LABEL = "inplane"
PI_SIGMA_S_PER_M = 1e-9
ELECTROSTATIC_FREQ_HZ = 1000.0
TRACE_H_NM = 100.0


def require_single_file(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {input_dir / pattern}, found {len(matches)}")
    return matches[0]


def parse_1x1_table(path: Path, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        try:
            values = [float(part) for part in line.split()]
        except ValueError:
            continue
        if len(values) == len(columns):
            rows.append(dict(zip(columns, values)))
    return pd.DataFrame(rows)


def parse_2x2_table(path: Path, geom_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    current_geom: dict[str, float] | None = None
    first_row: list[float] | None = None

    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        try:
            values = [float(part) for part in line.split()]
        except ValueError:
            continue

        if len(values) == len(geom_columns) + 2:
            current_geom = dict(zip(geom_columns, values[: len(geom_columns)]))
            first_row = values[len(geom_columns) :]
        elif len(values) == 2 and current_geom is not None and first_row is not None:
            row = current_geom.copy()
            row.update({"m11": first_row[0], "m12": first_row[1], "m21": values[0], "m22": values[1]})
            rows.append(row)
            first_row = None

    return pd.DataFrame(rows)


def add_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["width_key_nm"] = out["width_nm"].astype(float).round(9)
    out["gap_key"] = out["gap_ratio"].astype(float).round(9)
    out["pi_key_nm"] = out["pi_t_nm"].astype(float).round(9)
    return out


def fmt_token(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def case_name(width_nm: float, gap_ratio: float, pi_t_um: float, pi_sigma: float) -> str:
    return (
        f"w{fmt_token(width_nm)}nm_"
        f"r{fmt_token(gap_ratio)}_"
        f"pi{fmt_token(pi_t_um)}um_"
        f"sigma{fmt_token(pi_sigma)}Sm"
    )


def plot_coupling(master: pd.DataFrame, path: Path, plane_label: str, y_log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for (gap_ratio, pi_t_um), group in master.groupby(["gap_ratio", "pi_t_um"], sort=True):
        group = group.sort_values("width_nm")
        y = group["C_mut_pf_per_m"].clip(lower=1e-30) if y_log_scale else group["C_mut_pf_per_m"]
        ax.plot(group["width_nm"], y, marker="o", linewidth=1.5, label=f"G/LW={gap_ratio:g}, PI={pi_t_um:g} um")
    ax.set_xscale("log")
    if y_log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Trace width (nm)")
    ax.set_ylabel("C_mut (pF/m label)")
    ax.set_title(f"Large-sweep {plane_label} COMSOL C_mut")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_master(input_dir: Path, plane_label: str) -> pd.DataFrame:
    coupling_path = require_single_file(input_dir, "LW_largerange_*_sweep_C_1t*.txt")
    shunt_c_path = require_single_file(input_dir, "LW_largerange_*_sweep_C_2t*[Mm]utual.txt")
    shunt_g_path = require_single_file(input_dir, "LW_largerange_*_sweep_G_2t.txt")

    c_1t = parse_1x1_table(
        coupling_path,
        ["width_nm", "gap_ratio", "pi_t_inter_nm", "C_mut_F"],
    )
    c_1t["pi_t_nm"] = 8.0 * c_1t["pi_t_inter_nm"] + 3.0 * TRACE_H_NM
    c_2x2 = parse_2x2_table(
        shunt_c_path,
        ["width_nm", "gap_ratio", "pi_t_inter_nm"],
    )
    c_2x2["pi_t_nm"] = 8.0 * c_2x2["pi_t_inter_nm"] + 3.0 * TRACE_H_NM
    g_2x2 = parse_2x2_table(
        shunt_g_path,
        ["width_nm", "gap_ratio", "pi_t_inter_nm"],
    )
    g_2x2["pi_t_nm"] = 8.0 * g_2x2["pi_t_inter_nm"] + 3.0 * TRACE_H_NM

    c_1t = add_merge_keys(c_1t)
    c_2x2 = add_merge_keys(c_2x2)
    g_2x2 = add_merge_keys(g_2x2)

    merged = c_1t.merge(
        c_2x2.drop(columns=["width_nm", "gap_ratio", "pi_t_nm", "pi_t_inter_nm"]),
        on=["width_key_nm", "gap_key", "pi_key_nm"],
        how="inner",
        validate="one_to_one",
    ).merge(
        g_2x2.drop(columns=["width_nm", "gap_ratio", "pi_t_nm", "pi_t_inter_nm"]),
        on=["width_key_nm", "gap_key", "pi_key_nm"],
        how="inner",
        validate="one_to_one",
        suffixes=("_c", "_g"),
    )
    if len(merged) != len(c_1t):
        raise RuntimeError(f"Merged {len(merged)} rows, expected {len(c_1t)} rows from C_1t input")

    g11 = merged["m11_g"].astype(float)
    g12 = merged["m12_g"].astype(float)
    g21 = merged["m21_g"].astype(float)
    g22 = merged["m22_g"].astype(float)
    g_offdiag = (g12 + g21) / 2.0
    negative_offdiag = g_offdiag < 0

    rows: list[dict[str, object]] = []
    for rec, neg, gmut_raw, gg1, gg2, gmut in zip(
        merged.to_dict(orient="records"),
        negative_offdiag,
        g_offdiag,
        np.where(negative_offdiag, g11 + g_offdiag, g11),
        np.where(negative_offdiag, g22 + g_offdiag, g22),
        np.where(negative_offdiag, -g_offdiag, g_offdiag),
    ):
        width_nm = float(rec["width_nm"])
        gap_ratio = float(rec["gap_ratio"])
        pi_t_inter_nm = float(rec["pi_t_inter_nm"])
        pi_t_um = float(rec["pi_t_nm"]) / 1000.0
        name = case_name(width_nm, gap_ratio, pi_t_um, PI_SIGMA_S_PER_M)
        rows.append(
            {
                "case_name": name,
                "design_name": f"COMSOL_large_sweep_{plane_label}_{name}",
                "width_nm": width_nm,
                "gap_ratio": gap_ratio,
                "gap_nm": width_nm * gap_ratio,
                "pi_t_um": pi_t_um,
                "pi_t_inter_nm": pi_t_inter_nm,
                "trace_h_nm": TRACE_H_NM,
                "pi_sigma_s_per_m": PI_SIGMA_S_PER_M,
                "frequency_hz": ELECTROSTATIC_FREQ_HZ,
                "source_name": f"combined_comsol_large_sweep_1em9_{plane_label}",
                "source_name_coupling": f"Large-sweep {plane_label} one-terminal COMSOL C export",
                "source_file_coupling": coupling_path.name,
                "matrix_mode_coupling": f"comsol_large_sweep_{plane_label}_1terminal_cmut",
                "source_name_shunt": f"Large-sweep {plane_label} two-terminal COMSOL C matrix export",
                "source_file_shunt": shunt_c_path.name,
                "matrix_mode_shunt": f"comsol_large_sweep_{plane_label}_2x2_capacitance",
                "source_name_g": f"Large-sweep {plane_label} two-terminal COMSOL G matrix export",
                "source_file_g": shunt_g_path.name,
                "matrix_mode_g": f"comsol_large_sweep_{plane_label}_2x2_conductance",
                "shunt_mapping": "1terminal_cmut_plus_2x2_cap_diagonals_and_g_mapped",
                "g_matrix_mapping": "nodal_negative_offdiag_to_mutual_and_shunt" if neg else "branch_positive_offdiag",
                "model_variant": f"comsol_large_sweep_1em9_{plane_label}",
                "C_mut_pf_per_m": float(rec["C_mut_F"]) * 1.0e12,
                "C_mut_matrix_pf_per_m": float((rec["m12_c"] + rec["m21_c"]) / 2.0) * 1.0e12,
                "C_g1_pf_per_m": float(rec["m11_c"]) * 1.0e12,
                "C_g2_pf_per_m": float(rec["m22_c"]) * 1.0e12,
                "G_mut_mS_per_m": float(gmut) * 1.0e3,
                "G_mut_raw_mS_per_m": float(gmut_raw) * 1.0e3,
                "G_g1_mS_per_m": max(float(gg1) * 1.0e3, 0.0),
                "G_g2_mS_per_m": max(float(gg2) * 1.0e3, 0.0),
                "source_file_cmut": coupling_path.name,
                "source_file_cg": shunt_c_path.name,
                "source_file_g": shunt_g_path.name,
            }
        )

    return pd.DataFrame(rows).sort_values(["frequency_hz", "gap_ratio", "width_nm", "pi_t_um"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a standard LTspice master CSV from large-sweep COMSOL files.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help=f"COMSOL input directory. Default: {INPUT_DIR}")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR, help=f"Output directory. Default: {OUT_DIR}")
    parser.add_argument("--plane-label", default=DEFAULT_PLANE_LABEL, help="Label used in output names and metadata, e.g. inplane or outplane.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = args.input_dir
    outdir = args.outdir
    master_csv = outdir / "comsol_two_trace_master.csv"
    plane_label = str(args.plane_label)

    outdir.mkdir(parents=True, exist_ok=True)
    master = build_master(input_dir, plane_label)
    master.to_csv(master_csv, index=False)
    plot_coupling(master, outdir / f"comsol_large_sweep_{plane_label}_cmut_vs_lw.png", plane_label, y_log_scale=False)
    plot_coupling(master, outdir / f"comsol_large_sweep_{plane_label}_cmut_vs_lw_logy.png", plane_label, y_log_scale=True)

    note = f"""Large-sweep {plane_label} COMSOL master generation
================================================

Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input directory:
- {input_dir}

Mapping:
- C_mut_pf_per_m comes from the matched C_1t file in the input directory.
- C_mut_matrix_pf_per_m preserves the average off-diagonal from the matched C_2t_Mutual file.
- C_g1_pf_per_m and C_g2_pf_per_m come from the 2x2 capacitance matrix diagonals.
- PI sigma is fixed to {PI_SIGMA_S_PER_M:g} S/m for every row.
- COMSOL pi_t_inter is converted to the previous pi_t convention with pi_t = 8*pi_t_inter + 3*trace_h.
- trace_h is fixed to {TRACE_H_NM:g} nm.
- The resulting pi_t is stored in um in the master CSV; raw pi_t_inter_nm is preserved.
- Conductance mapping follows the existing leaky flow: negative off-diagonal conductance is treated as a nodal matrix.

Rows written: {len(master)}
LW values kept: {', '.join(f'{v:g}' for v in sorted(master['width_nm'].unique()))}
G/LW values kept: {', '.join(f'{v:g}' for v in sorted(master['gap_ratio'].unique()))}
PI values kept: {', '.join(f'{v:g}' for v in sorted(master['pi_t_um'].unique()))}
PI-inter values kept: {', '.join(f'{v:g}' for v in sorted(master['pi_t_inter_nm'].unique()))} nm
"""
    (outdir / "GENERATION_NOTE.txt").write_text(note, encoding="utf-8")

    print(f"Master CSV: {master_csv.resolve()}")
    print(f"Rows: {len(master)}")
    print(f"Output dir: {outdir.resolve()}")


if __name__ == "__main__":
    main()
