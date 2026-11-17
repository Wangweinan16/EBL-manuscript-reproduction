"""Validate the plotted values against saved SpikeInterface sorting/template arrays."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    with (ROOT / "units.csv").open(newline="") as stream:
        units = list(csv.DictReader(stream))
    with (ROOT / "sessions.csv").open(newline="") as stream:
        sessions = list(csv.DictReader(stream))
    for session in sessions:
        animal, layout = session["animal"], int(session["layout"])
        group = {int(row["unit_id"]): row for row in units if row["animal"] == animal and int(row["layout"]) == layout}
        source = ROOT / "sorting" / animal / f"layout_{layout:03d}" / "sorting.npz"
        with np.load(source, allow_pickle=False) as data:
            ids = data["unit_ids"]
            templates = data["templates_average"]
            counts = np.bincount(data["spikes"]["unit_index"], minlength=len(ids))
            assert set(group) == set(ids)
            for i, unit_id in enumerate(ids):
                row = group[unit_id]
                ptp = np.ptp(templates[i], axis=0)
                best = int(np.argmax(ptp))
                assert best == int(row["peak_channel_index"])
                np.testing.assert_allclose(ptp[best], float(row["peak_p2p_uv"]),rtol=1e-6)
                assert int(np.sum(ptp >= 0.2*ptp[best])) == int(row["localized_electrodes_20pct"])
                assert counts[i] == int(row["spike_count"])
        assert sum(r["status"]=="accepted" for r in group.values()) == int(session["accepted_count"])
    print(f"Validated {len(units)} units across {len(sessions)} animal/layout sessions.")

if __name__ == "__main__":
    main()
