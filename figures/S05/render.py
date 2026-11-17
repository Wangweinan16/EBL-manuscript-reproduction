"""Reproduce S5A and check the observed S5B CCG from cached unit spike trains."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent

def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("svg", {"format": "svg"}),
        ("pdf", {"format": "pdf"}),
    ):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        print(f"saved {path}", flush=True)
    plt.close(fig)


def pair_channel_cache_indices(data: dict[str, np.ndarray], pair: tuple[int, int]) -> np.ndarray:
    selected_units = data["selected_units"].astype(int).tolist()
    unit_rows = [selected_units.index(unit_id) for unit_id in pair]
    return np.unique(data["neighbor_cache_indices"][unit_rows].ravel())


def waveform_xy(
    waveform: np.ndarray,
    channel_xy: np.ndarray,
    sampling_frequency: float,
    *,
    time_width_um: float,
    uv_to_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_ms = np.arange(waveform.size, dtype=float) * 1000.0 / sampling_frequency
    x = channel_xy[0] + (time_ms / time_ms[-1] - 0.5) * time_width_um
    y = channel_xy[1] + waveform * uv_to_um
    return x, y


def draw_probe_waveforms(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    units: tuple[int, ...],
    channel_cache_indices: np.ndarray,
    *,
    show_distributions: bool,
    time_width_um: float,
    uv_to_um: float,
) -> None:
    selected_units = data["selected_units"].astype(int).tolist()
    channel_locations = data["channel_locations"]
    cached_channel_indices = data["cached_channel_indices"].astype(int)
    sampling_frequency = float(data["sampling_frequency"])
    original_channel_indices = cached_channel_indices[channel_cache_indices]
    plotted_locations = channel_locations[original_channel_indices]
    y_min = float(plotted_locations[:, 1].min() - 65)
    y_max = float(plotted_locations[:, 1].max() + 65)
    visible_contacts = (channel_locations[:, 1] >= y_min) & (channel_locations[:, 1] <= y_max)

    ax.plot([-39, -39, 39, 39], [y_min, y_max, y_max, y_min], color="black", lw=0.45)
    ax.scatter(
        channel_locations[visible_contacts, 0],
        channel_locations[visible_contacts, 1],
        s=5,
        color="black",
        linewidth=0,
        zorder=5,
    )

    for unit_id in units:
        unit_row = selected_units.index(unit_id)
        color = data["colors_rgba"][unit_row]
        own_neighbors = set(data["neighbor_cache_indices"][unit_row].astype(int).tolist())
        for cache_index in channel_cache_indices:
            cache_index = int(cache_index)
            if cache_index not in own_neighbors:
                continue
            channel_index = int(cached_channel_indices[cache_index])
            channel_xy = channel_locations[channel_index]
            if show_distributions:
                count = int(data["sampled_counts"][unit_row])
                for waveform in data["sampled_waveforms"][unit_row, :count, :, cache_index]:
                    x, y = waveform_xy(
                        waveform,
                        channel_xy,
                        sampling_frequency,
                        time_width_um=time_width_um,
                        uv_to_um=uv_to_um,
                    )
                    ax.plot(x, y, color=color, lw=0.22, alpha=0.075, zorder=1)
            mean_waveform = data["templates"][unit_row, :, cache_index]
            x, y = waveform_xy(
                mean_waveform,
                channel_xy,
                sampling_frequency,
                time_width_um=time_width_um,
                uv_to_um=uv_to_um,
            )
            ax.plot(x, y, color=color, lw=0.9, alpha=0.95, zorder=3)

        location = data["unit_locations"][unit_row]
        ax.scatter(
            [location[0]],
            [location[1]],
            s=32,
            facecolor=mcolors.to_rgba(color, 0.22),
            edgecolor=color,
            linewidth=0.8,
            zorder=6,
            clip_on=False,
            label=f"Unit {unit_id}",
        )

    ax.set_xlim(-58, 68)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance (um)", fontsize=7)
    ax.set_ylabel("Distance (um)", fontsize=7)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(0.5)
    ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))

    scale_x = 52
    scale_y = y_min + 45
    one_ms_width = time_width_um / 3.0
    ax.plot([scale_x - one_ms_width, scale_x], [scale_y, scale_y], color="black", lw=0.7)
    ax.plot(
        [scale_x, scale_x],
        [scale_y, scale_y + 50 * uv_to_um],
        color="black",
        lw=0.7,
    )
    ax.text(scale_x - one_ms_width / 2, scale_y - 9, "1 ms", ha="center", va="top", fontsize=5)
    ax.text(scale_x + 8, scale_y + 25 * uv_to_um, "50 uV", ha="left", va="center", fontsize=5, rotation=90)


def plot_pair_waveforms(data: dict[str, np.ndarray], output_dir: Path, pair: tuple[int, int]) -> None:
    channels = pair_channel_cache_indices(data, pair)
    original_indices = data["cached_channel_indices"][channels.astype(int)].astype(int)
    y_span = float(np.ptp(data["channel_locations"][original_indices, 1])) + 130
    height = float(np.clip(2.8 + y_span / 260.0, 3.2, 7.5))
    fig, ax = plt.subplots(figsize=(2.65, height))
    draw_probe_waveforms(
        ax,
        data,
        pair,
        channels,
        show_distributions=True,
        time_width_um=28.0,
        uv_to_um=0.14,
    )
    ax.set_title(f"Putative connection: unit {pair[0]} to unit {pair[1]}", fontsize=7)
    fig.subplots_adjust(left=0.22, right=0.74, bottom=0.10, top=0.94)
    save_figure(fig, output_dir / "pair_waveform_distributions", f"unit{pair[0]}_to_unit{pair[1]}_neighbor_waveforms")

def bounded_jitter(
    spikes_s: np.ndarray,
    *,
    duration_s: float,
    jitter_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    jitter_s = jitter_ms / 1000.0
    jittered = spikes_s + rng.uniform(-jitter_s, jitter_s, spikes_s.size)
    outside = (jittered < 0.0) | (jittered > duration_s)
    while np.any(outside):
        jittered[outside] = spikes_s[outside] + rng.uniform(
            -jitter_s, jitter_s, int(np.sum(outside))
        )
        outside = (jittered < 0.0) | (jittered > duration_s)
    jittered.sort()
    return jittered

def pair_counts(a, b):
    # Match the detector's floating-point subtraction in seconds, including ties on bin edges.
    edges = np.arange(-4., 4.5, .5)
    left = np.searchsorted(b, a - .004, side='left')
    right = np.searchsorted(b, a + .004, side='right')
    n = right - left
    n_pairs = np.sum(n)
    starts = np.repeat(np.cumsum(n) - n, n)
    indexes = np.repeat(left, n) + np.arange(n_pairs) - starts
    lag_ms = (b[indexes] - np.repeat(a, n)) * 1000.
    return np.histogram(lag_ms, bins=edges)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with np.load(HERE / 'units_29_30.npz', allow_pickle=False) as load:
        data = dict(load)
    plot_pair_waveforms(data, args.out, (29, 30))
    with np.load(HERE / 'ccg.npz', allow_pickle=False) as load:
        ccg = dict(load)
    fs = float(data['sampling_frequency'])
    a = data['spike_samples_29']
    b = data['spike_samples_30']
    a, b = a.astype(float) / fs, b.astype(float) / fs
    counts = pair_counts(a, b)
    np.testing.assert_array_equal(counts, ccg['observed_ccg'])
    rng = np.random.default_rng(int(ccg['pair_seed']))
    null = np.zeros((1000, 16), dtype=np.float32)
    for repeat in range(len(null)):
        ja = bounded_jitter(a, duration_s=1940.5642333333333, jitter_ms=10., rng=rng)
        jb = bounded_jitter(b, duration_s=1940.5642333333333, jitter_ms=10., rng=rng)
        null[repeat] = pair_counts(ja, jb)
    np.testing.assert_array_equal(null.mean(axis=0), ccg['jitter_mean'])
    np.testing.assert_array_equal(np.percentile(null, 99, axis=0), ccg['jitter_ci99'])
    np.testing.assert_allclose(null.std(axis=0, ddof=1), ccg['jitter_std'])
    unit_a, unit_b = 29, 30
    peak_lag_ms, peak_z = float(ccg['peak_lag_ms']), float(ccg['peak_z'])
    peak_count = counts[int(ccg['peak_index'])]
    observed, lags_ms = counts, ccg['lags_ms']
    jitter_mean, jitter_ci99 = null.mean(axis=0), np.percentile(null, 99, axis=0)
    run_dir, stem = args.out, "S05B_CCG"
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.bar(lags_ms, observed, width=.5, color="0.25", label="Observed CCG")
    ax.plot(lags_ms, jitter_mean, color="#1f77b4", linewidth=0.8, label="Jitter mean")
    ax.plot(lags_ms, jitter_ci99, "--", color="#d62728", linewidth=1.0, label="99% jitter CI")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.scatter([peak_lag_ms], [peak_count], s=12, color="#d62728", zorder=4)
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel("Lag, unit B - unit A (ms)", fontsize=8)
    ax.set_ylabel("CCG count", fontsize=8)
    status_text = "PASS"
    direction = "29->30"
    ax.set_title(
        f"Units {unit_a}/{unit_b}: peak {peak_lag_ms:.2f} ms, Z={peak_z:.2f}\n"
        f"{direction}; {status_text}",
        fontsize=8,
    )
    ax.tick_params(labelsize=7, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(run_dir / f"{stem}.svg", format="svg")
    fig.savefig(run_dir / f"{stem}.pdf", format="pdf")
    plt.close(fig)


if __name__ == '__main__':
    main()
