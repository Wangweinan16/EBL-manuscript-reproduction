"""Replot the Figure 3B trace at its published placement and validate it against raw ADC samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent


def plot(out: Path) -> None:
    meta = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    drawn = pd.read_csv(HERE / "voltage_path.csv")
    arrows = pd.read_csv(HERE / "curated_spike_arrows.csv")
    train = pd.read_csv(HERE / "stimulus_train.csv")
    with np.load(HERE / "acquisition_window.npz", allow_pickle=False) as source:
        adc = source["adc_counts"].astype(float)
        start = int(source["window_start_frame"])
        fs = float(source["fs_hz"])
    gain = (meta["analog_max"] - meta["analog_min"]) / (meta["digital_max"] - meta["digital_min"])
    voltage_uv = meta["analog_min"] + (adc - meta["digital_min"]) * gain
    transform = meta["source_vector_transform"]
    x_raw = transform["x_pt_at_window_start"] + np.arange(len(adc)) * transform["x_pt_per_sample"]
    y_raw = transform["y_pt_at_zero_uV"] + voltage_uv * transform["y_pt_per_uV"]
    drawn_sample = np.rint((drawn.page_x_pt-x_raw[0])/transform["x_pt_per_sample"]).astype(int)
    np.testing.assert_allclose(drawn.page_y_pt, y_raw[drawn_sample], rtol=0, atol=0.0006)
    arrow_frame = start + (arrows.page_x_pt-x_raw[0])/transform["x_pt_per_sample"]
    np.testing.assert_allclose(arrow_frame,arrows.sample_frame,atol=0.007,rtol=0)
    visible = train[train.visible_in_plate]
    assert list(visible.pulse_number) == list(range(4,10))
    np.testing.assert_array_equal(np.diff(train.sample_frame),np.full(49,600))
    np.testing.assert_equal(float(fs),30000.)
    out.mkdir(parents=True,exist_ok=True)
    # Right/up page coordinates are in PDF points. The clipping bounds retain the selected six pulses.
    x, y, w, h = meta["clip_box_pdf_points"]
    for mode, xs, ys in [("source_vector",drawn.page_x_pt,drawn.page_y_pt),
                         ("acquisition",x_raw,y_raw)]:
        fig,ax=plt.subplots(figsize=(w/72,h/72))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.set(xlim=(x,x+w),ylim=(y,y+h))
        ax.axis("off")
        for left,right in meta["artifact_bands_pdf_bounds_x_pt"]:
            ax.axvspan(left,right,color="#e7e7e7",linewidth=0)
        for marker in meta["visible_pulse_marker_x_pt"]:
            ax.axvline(marker,color="#547ca6",lw=0.45,dashes=(1.665,0.72))
        ax.plot(xs,ys,color="#222222",lw=0.4,solid_capstyle="projecting")
        for arrow in arrows.itertuples():
            ax.plot([arrow.page_x_pt]*2,[arrow.arrow_base_y_pt,arrow.arrow_base_y_pt+9.805],
                    color="#da2929",lw=0.45)
            ax.add_patch(Polygon([[arrow.page_x_pt,arrow.arrow_base_y_pt+9.805],
                                  [arrow.page_x_pt-1.162,arrow.arrow_base_y_pt+6.866],
                                  [arrow.page_x_pt+1.162,arrow.arrow_base_y_pt+6.866]],
                                 closed=True,color="#da2929",linewidth=0))
        for ext in ("svg","png"):
            fig.savefig(out/f"Figure3B_{mode}.{ext}",dpi=300)
        plt.close(fig)
    # A lossless numeric export preserves every acquired frame across the full original 220 ms crop.
    frame = start + np.arange(len(adc))
    pd.DataFrame({"sample_frame":frame,"ms_from_first_pulse":(frame-train.sample_frame.iloc[0])*1000/fs,
                  "voltage_uV":voltage_uv}).to_csv(out/"Figure3B_calibrated_samples.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    plot(parser.parse_args().out)
