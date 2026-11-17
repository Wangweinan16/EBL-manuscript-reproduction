from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
EVENTS_PATH = ROOT / "data/events.csv"
LEDGER_PATH = ROOT / "data/sessions.csv"
OUT = ROOT.parents[1] / "outputs" / "3E"
DATA = OUT / "intermediate_data"
FIG = OUT / "figures"

ANIMALS = ["EBL2", "EBL6", "EBL7", "EBL8", "EBL9"]
CURRENTS = [3.0, 4.0, 5.0]
COLORS = {3.0: "#0072B2", 4.0: "#D55E00", 5.0: "#009E73"}

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    }
)


def load_channel_positions(animal: str) -> dict[int, np.ndarray]:
    path = ROOT / "data" / "geometry" / (animal + ".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe = payload["probes"][0]
    positions = np.asarray(probe["contact_positions"], dtype=float)
    device_indices = probe.get("device_channel_indices")
    if device_indices is None:
        device_indices = list(range(len(positions)))
    mapping = {int(index) + 1: positions[i, :2] for i, index in enumerate(device_indices)}
    if len(mapping) != 128:
        raise ValueError(f"{animal}: expected 128 mapped contacts, found {len(mapping)}")
    return mapping


def build_pair_table(events: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    position_maps = {animal: load_channel_positions(animal) for animal in ANIMALS}
    rows: list[dict] = []
    for session in ledger.itertuples(index=False):
        channels = sorted(int(value) for value in str(session.channels).split(","))
        session_events = events[
            (events["animal"] == session.animal)
            & (events["session_date"] == session.session_date)
        ]
        for current in CURRENTS:
            current_events = session_events[session_events["current_uA"] == current]
            units_by_channel = {
                channel: set(
                    current_events.loc[current_events["stim_channel"] == channel, "unit_id"].astype(int)
                )
                for channel in channels
            }
            for i, channel_a in enumerate(channels):
                for channel_b in channels[i + 1 :]:
                    units_a = units_by_channel[channel_a]
                    units_b = units_by_channel[channel_b]
                    union = units_a | units_b
                    intersection = units_a & units_b
                    loc_a = position_maps[session.animal][channel_a]
                    loc_b = position_maps[session.animal][channel_b]
                    rows.append(
                        {
                            "animal": session.animal,
                            "session_date": session.session_date,
                            "session_key": f"{session.animal}_{session.session_date}",
                            "current_uA": current,
                            "channel_a": channel_a,
                            "channel_b": channel_b,
                            "distance_um": float(np.linalg.norm(loc_a - loc_b)),
                            "n_activated_a": len(units_a),
                            "n_activated_b": len(units_b),
                            "n_intersection": len(intersection),
                            "n_union": len(union),
                            "both_channels_responsive": bool(units_a and units_b),
                            "union_nonempty": bool(union),
                            "common_fraction_percent": (
                                100.0 * len(intersection) / len(union) if union else np.nan
                            ),
                            "activation_selectivity_percent": (
                                100.0 * (1.0 - len(intersection) / len(union)) if union else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def model(distance: np.ndarray, amplitude: float, rate: float, floor: float) -> np.ndarray:
    return 100.0 - (amplitude * np.exp(-rate * distance) + floor)


def fit_curve(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby("distance_um", as_index=False)["activation_selectivity_percent"].mean()
    params, _ = curve_fit(
        model,
        grouped["distance_um"].to_numpy(),
        grouped["activation_selectivity_percent"].to_numpy(),
        p0=[50.0, 0.01, 20.0],
        bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]),
        maxfev=20000,
    )
    x = np.linspace(0.0, 600.0, 401)
    return params, np.clip(model(x, *params), 0.0, 100.0)


def cluster_bootstrap(frame: pd.DataFrame, n_boot: int = 500) -> tuple[np.ndarray, np.ndarray, int]:
    sessions = frame["session_key"].unique()
    rng = np.random.default_rng(20260920)
    predictions = []
    x = np.linspace(0.0, 600.0, 401)
    for _ in range(n_boot):
        pieces = []
        for draw, session in enumerate(rng.choice(sessions, size=len(sessions), replace=True)):
            piece = frame[frame["session_key"] == session].copy()
            piece["bootstrap_session"] = draw
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        grouped = sample.groupby(["bootstrap_session", "distance_um"], as_index=False)[
            "activation_selectivity_percent"
        ].mean()
        try:
            params, _ = curve_fit(
                model,
                grouped["distance_um"].to_numpy(),
                grouped["activation_selectivity_percent"].to_numpy(),
                p0=[50.0, 0.01, 20.0],
                bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        predictions.append(np.clip(model(x, *params), 0.0, 100.0))
    if not predictions:
        return np.full_like(x, np.nan), np.full_like(x, np.nan), 0
    values = np.asarray(predictions)
    return np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0), len(values)


def summarize_and_plot(pair_table: pd.DataFrame, mode: str, mask: pd.Series) -> pd.DataFrame:
    selected = pair_table[mask & (pair_table["distance_um"] < 600)].copy()
    selected.to_csv(DATA / f"Fig3g_pair_level_{mode}.csv", index=False)
    x = np.linspace(0.0, 600.0, 401)
    stats_rows = []
    fig, ax = plt.subplots(figsize=(2.16, 2.25))
    for current in CURRENTS:
        frame = selected[selected["current_uA"] == current]
        rho, p_value = spearmanr(frame["distance_um"], frame["activation_selectivity_percent"])
        params, prediction = fit_curve(frame)
        lower, upper, n_success = cluster_bootstrap(frame)
        color = COLORS[current]
        ax.scatter(
            frame["distance_um"],
            frame["activation_selectivity_percent"],
            color=color,
            alpha=0.18,
            s=2.2,
            edgecolors="none",
            linewidths=0,
            zorder=2,
        )
        if n_success:
            ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(x, prediction, color=color, linewidth=1.1, label=f"{current:g} μA", zorder=3)
        stats_rows.append(
            {
                "mode": mode,
                "current_uA": current,
                "n_pair_session_observations": len(frame),
                "n_sessions": frame["session_key"].nunique(),
                "n_animals": frame["animal"].nunique(),
                "spearman_rho": rho,
                "spearman_p_value_naive": p_value,
                "fit_amplitude": params[0],
                "fit_rate_per_um": params[1],
                "fit_floor_common_percent": params[2],
                "cluster_bootstrap_successes": n_success,
            }
        )
    ax.set_xlabel("Distance between channels (μm)")
    ax.set_ylabel("Activation selectivity (%)")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", width=0.5, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.legend(frameon=False, fontsize=6, handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"Fig3g_five_animals_ten_sessions_{mode}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS_PATH, dtype={"session_date": str})
    ledger = pd.read_csv(LEDGER_PATH, dtype={"session_date": str})
    ledger = ledger[
        ledger["animal"].isin(ANIMALS)
        & ledger["ready_for_offset_corrected_reanalysis"].astype(bool)
    ].copy()
    if len(ledger) != 10:
        raise ValueError(f"Expected 10 validated sessions, found {len(ledger)}")
    pair_table = build_pair_table(events, ledger)
    pair_table.to_csv(DATA / "Fig3g_all_channel_pairs_complete_grid.csv", index=False)

    session_rows = []
    session_analysis = pair_table[
        pair_table["union_nonempty"] & (pair_table["distance_um"] < 600)
    ]
    for (animal, session_date, current), frame in session_analysis.groupby(
        ["animal", "session_date", "current_uA"], sort=True
    ):
        rho, p_value = spearmanr(
            frame["distance_um"], frame["activation_selectivity_percent"]
        )
        session_rows.append(
            {
                "animal": animal,
                "session_date": session_date,
                "current_uA": current,
                "n_pair_observations": len(frame),
                "spearman_rho": rho,
                "spearman_p_value_naive": p_value,
                "mean_activation_selectivity_percent": frame[
                    "activation_selectivity_percent"
                ].mean(),
            }
        )
    pd.DataFrame(session_rows).to_csv(
        DATA / "Fig3g_session_level_association_summary.csv", index=False
    )

    primary_stats = summarize_and_plot(pair_table, "union_nonempty", pair_table["union_nonempty"])
    historical_stats = summarize_and_plot(
        pair_table, "both_channels_responsive", pair_table["both_channels_responsive"]
    )
    stats_table = pd.concat([primary_stats, historical_stats], ignore_index=True)
    stats_table.to_csv(DATA / "Fig3g_fit_and_association_summary.csv", index=False)

    completeness = (
        pair_table.groupby(["animal", "session_date", "current_uA"], as_index=False)
        .agg(
            total_channel_pairs=("channel_a", "size"),
            union_nonempty_pairs=("union_nonempty", "sum"),
            both_channels_responsive_pairs=("both_channels_responsive", "sum"),
        )
    )
    completeness.to_csv(DATA / "Fig3g_pair_inclusion_summary_by_session_current.csv", index=False)

    manifest = {
        "run_id": OUT.name,
        "purpose": "Selected five-animal, ten-session reproduction of November Figure 3E",
        "source_activation_events": "data/events.csv",
        "source_session_ledger": "data/sessions.csv",
        "animals": ANIMALS,
        "sessions": ledger[["animal", "session_date"]].to_dict("records"),
        "currents_plotted_uA": CURRENTS,
        "activation_criterion_inherited_from_source": {
            "test": "two-sided Welch t-test across ten stimulation and ten 2-s baseline rates",
            "threshold": "p <= 0.001 and mean stimulation rate > mean baseline rate",
        },
        "primary_pair_inclusion": "All 12-channel pairs with a nonempty union of activated units. Both-zero pairs are excluded because Jaccard overlap is undefined.",
        "sensitivity_pair_inclusion": "Both stimulation channels must activate at least one unit, matching the inherited Figure 3g construction.",
        "selectivity": "100 * (1 - intersection/union) for activated-unit sets",
        "fit": "Bounded three-parameter exponential fitted after averaging exact repeated distances",
        "confidence_band": "95% percentile band from 500 session-cluster bootstrap resamples",
        "status": "selected for November Figure 3E on 2026-09-21; centralized through sync_selected_main_figure_panels.py",
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(pair_table)} complete-grid pair/current rows to {OUT}")
    print(stats_table.to_string(index=False))


if __name__ == "__main__":
    main()
