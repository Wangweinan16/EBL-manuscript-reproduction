from __future__ import annotations

import hashlib
import io
import math
from collections import OrderedDict
from pathlib import Path
import re
from zipfile import ZipFile
import pandas as pd
import numpy as np
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
NT_RE = re.compile(r"\bNT\s*(?P<ch>\d+)\s*,\s*Channel\s*\d+\s*:\s*(?P<imp>-?\d+(?:\.\d+)?)", re.IGNORECASE)

CH_RE = re.compile(r"\bCH(?P<ch>\d+)\b", re.IGNORECASE)

COUNTIF_THRESHOLD_RE = re.compile(r"COUNTIF\([^,]+,\s*\"<(?P<thr>\d+(?:\.\d+)?)\"\)", re.IGNORECASE)

SUMMARY_LABELS = {"ref", "shorted", "connected", "single", "yield 1", "yield 2"}

LOCAL_REF_CHANNELS = {9, 12, 16, 18, 20, 21, 23, 24, 28, 30, 31, 64, 65, 67, 69, 95, 98, 105}

def as_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None

def numeric_cells_after(ws, row_idx: int, start_col: int, max_col: int) -> list[float]:
    vals = []
    for col in range(start_col, max_col + 1):
        val = as_float(ws.cell(row_idx, col).value)
        if val is not None:
            vals.append(val)
    return vals

def extract_latest_channel_values_xlsx(path: Path) -> tuple[OrderedDict[int, float], int, int, str]:
    try:
        wb = load_workbook(path, data_only=True, read_only=False)
    except Exception as exc:
        return OrderedDict(), 0, 0, f"xlsx_open_error:{type(exc).__name__}"

    latest: OrderedDict[int, float] = OrderedDict()
    row_counter = 0
    measurement_event_count = 0
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for row in ws.iter_rows():
            row_counter += 1
            for cell in row:
                text = cell.value if isinstance(cell.value, str) else ""
                if not text:
                    continue
                nt_match = NT_RE.search(text)
                if nt_match:
                    ch = int(nt_match.group("ch"))
                    vals = numeric_cells_after(ws, cell.row, cell.column + 1, min(ws.max_column, cell.column + 4))
                    imp = vals[0] if vals else as_float(nt_match.group("imp"))
                    if imp is not None and imp > 0:
                        measurement_event_count += 1
                        latest[ch] = imp
                    continue
                ch_match = CH_RE.fullmatch(text.strip())
                if ch_match:
                    ch = int(ch_match.group("ch"))
                    vals = numeric_cells_after(ws, cell.row, cell.column + 1, min(ws.max_column, cell.column + 4))
                    imp = vals[0] if vals else None
                    if imp is not None and imp > 0:
                        measurement_event_count += 1
                        latest[ch] = imp
    return latest, row_counter, measurement_event_count, "ok"

def extract_summary_and_formulas_xlsx(path: Path) -> dict[str, object]:
    try:
        wb_f = load_workbook(path, data_only=False, read_only=False)
        wb_v = load_workbook(path, data_only=True, read_only=False)
    except Exception:
        return {}
    out: dict[str, object] = {}
    formula_out: dict[str, str] = {}
    for sheet_name in wb_f.sheetnames:
        ws_f = wb_f[sheet_name]
        ws_v = wb_v[sheet_name]
        for row in range(1, min(ws_f.max_row, 40) + 1):
            for col in range(1, min(ws_f.max_column, 10) + 1):
                label_raw = ws_f.cell(row, col).value
                if not isinstance(label_raw, str):
                    continue
                label = label_raw.strip().lower()
                if label not in SUMMARY_LABELS:
                    continue
                for val_col in [col + 1, col + 2]:
                    if val_col > ws_f.max_column:
                        continue
                    data_value = as_float(ws_v.cell(row, val_col).value)
                    formula_value = ws_f.cell(row, val_col).value
                    if data_value is None and formula_value in (None, ""):
                        continue
                    out[label] = data_value if data_value is not None else ""
                    formula_out[label] = str(formula_value) if formula_value is not None else ""
                    break

    for key in SUMMARY_LABELS:
        out[f"formula_{key.replace(' ', '')}"] = formula_out.get(key, "")
    short_match = COUNTIF_THRESHOLD_RE.search(str(out.get("formula_shorted", "")))
    conn_match = COUNTIF_THRESHOLD_RE.search(str(out.get("formula_connected", "")))
    out["short_threshold_ohm"] = float(short_match.group("thr")) if short_match else ""
    out["connected_threshold_ohm"] = float(conn_match.group("thr")) if conn_match else ""
    return out

def ref_mode(summary: dict[str, object]) -> tuple[bool, str]:
    ref = as_float(summary.get("ref"))
    if ref is not None and int(round(ref)) == len(LOCAL_REF_CHANNELS):
        return True, "explicit_workbook_ref_18"
    if ref is not None:
        return False, f"explicit_workbook_ref_{int(round(ref))}"
    return False, "workbook_ref_missing_assume_no_ref"

def main():
    index = pd.read_csv(HERE/'workbook_index.csv')
    reference = pd.read_csv(HERE/'S4_wafer_specific_channel_classification.csv')
    devices = pd.read_csv(HERE/'S4_wafer_specific_device_yield_and_impedance_summary.csv').set_index('device_id_norm')
    verified = 0
    with ZipFile(HERE/'source_workbooks.zip') as archive:
        assert set(archive.namelist()) == set(index.zip_entry) and len(index)==167
        for row in index.itertuples():
            payload = archive.read(row.zip_entry)
            assert hashlib.sha256(payload).hexdigest() == row.sha256
            channels, _, _, status = extract_latest_channel_values_xlsx(io.BytesIO(payload))
            assert status == 'ok', (row.device,status)
            use_ref, _ = ref_mode(extract_summary_and_formulas_xlsx(io.BytesIO(payload)))
            assert bool(devices.loc[row.device, 'refs_enabled']) == use_ref
            ana = {k:v for k,v in channels.items() if 1<=k<=128 and v>0 and not(use_ref and k in LOCAL_REF_CHANNELS)}
            expected=reference[reference.device_id_norm==row.device].set_index('channel_number').impedance_ohm.to_dict()
            assert ana == expected, row.device
            verified += len(ana)
    print(f'{len(index)} original workbook hashes and {verified} nonreference channel measurements matched.')

if __name__=='__main__':
    main()
