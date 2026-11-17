import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
def _isi_rc_params(font_family: str, text_as_path: bool) -> dict:
    if font_family == "arial":
        family = "Arial"
        sans_serif = ["Arial"]
    elif font_family == "sans":
        family = "sans-serif"
        sans_serif = ["DejaVu Sans", "Arial", "Helvetica"]
    else:
        raise ValueError(f"Unknown font family: {font_family}")
    return {
        "font.family": family,
        "font.sans-serif": sans_serif,
        "svg.fonttype": "path" if text_as_path else "none",
        "pdf.fonttype": 3 if text_as_path else 42,
        "ps.fonttype": 3 if text_as_path else 42,
    }


def draw_compact_isi_grid(
    waveform_root: Path,
    output_dir: Path,
    unit_id: int,
    *,
    filename_suffix: str = "_compact_copy",
) -> None:
    """Draw a compact normalized 10-day ISI grid for one tracked unit."""
    with matplotlib.rc_context(_isi_rc_params("arial", False)):
        isi_by_day = []
        with np.load(waveform_root / "isi_spike_frames.npz") as data:
            fs = float(data["sampling_frequency"])
            for day in range(1, 11):
                frames = data[f"day{day}_unit{unit_id}"]
                isi_by_day.append(np.diff(frames / fs) * 1000.0)

        bins = np.arange(0, 151, 5)
        fig, axes = plt.subplots(2, 5, figsize=(1.7, 0.8), sharex=False, sharey=False)

        for i, ax in enumerate(axes.flat):
            isi_ms = isi_by_day[i]
            isi_ms = isi_ms[(isi_ms >= 0) & (isi_ms <= 150)]
            if len(isi_ms):
                weights = np.ones_like(isi_ms, dtype=float) / len(isi_ms)
                ax.hist(isi_ms, bins=bins, weights=weights, color="black", edgecolor="black", linewidth=0)

            ax.set_xlim(-2.5, 150)
            ax.set_ylim(bottom=0)
            ax.autoscale(axis="y", tight=False)
            ax.text(0.95, 0.88, f"D{i + 1}", transform=ax.transAxes, fontsize=4.5, ha="right", va="top")

            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_linewidth(0.5)

            if i != 5:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
            else:
                ax.set_xticks([0, 150])
                ax.tick_params(axis="both", labelsize=4.5, width=0.5, length=1.5, pad=0.5)
                ax.set_xlabel("ISI (ms)", fontsize=6, labelpad=0.5)
                ax.set_ylabel("Fraction of ISIs", fontsize=6, labelpad=0.5)
                ax.spines["left"].set_position(("outward", 1.5))

        fig.subplots_adjust(left=0.18, right=0.99, bottom=0.30, top=0.96, wspace=0.12, hspace=0.20)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_base = output_dir / f"Fig3d_e_unit{unit_id}_ISI_10days{filename_suffix}"
        fig.savefig(out_base.with_suffix(".svg"), format="svg", transparent=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=600, transparent=True)
        fig.savefig(out_base.with_suffix(".pdf"), format="pdf", transparent=True)
        plt.close(fig)



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Plot Fig 2E ISI distributions. The top histogram is unit 29 and the bottom histogram is unit 31.")
    parser.add_argument("--unit", type=int, choices=[10,21,29,30,31,88], help="Optional individual unit. Default: both manuscript histograms.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args=parser.parse_args()
    for unit_id in ([args.unit] if args.unit is not None else [29, 31]):
        draw_compact_isi_grid(Path(__file__).resolve().parent / "data", args.output_dir, unit_id)
