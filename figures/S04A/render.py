from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parent
WAFERS = ("62", "71", "72")
DEVICE_MEDIAN_MIN_KOHM = 80.0
DEVICE_MEDIAN_MAX_KOHM = 200.0
DEVICE_GOOD_LOWER_MULTIPLIER = 0.70
DEVICE_GOOD_UPPER_MULTIPLIER = 1.50
MIN_DEVICE_YIELD = 75.0
N_BEST_YIELD_DEVICES = 100
DISPLAY_MIN_KOHM = 20.0
DISPLAY_MAX_KOHM = 1000.0
RANDOM_SEED = 128


def configure_style() -> None:
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
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    configure_style()
    DATA.mkdir(parents=True, exist_ok=True)

    devices = pd.read_csv(
        SOURCE / "S4_wafer_specific_device_yield_and_impedance_summary.csv",
        dtype={"wafer_norm": str, "device_id_norm": str},
    )
    channels = pd.read_csv(
        SOURCE / "S4_wafer_specific_channel_classification.csv",
        dtype={"wafer_norm": str, "device_id_norm": str},
    )
    devices = devices[devices["wafer_norm"].isin(WAFERS)].copy()
    channels = channels[channels["wafer_norm"].isin(WAFERS)].copy()
    devices["pooled_device_id"] = "W" + devices["wafer_norm"] + ":" + devices["device_id_norm"]
    channels["pooled_device_id"] = "W" + channels["wafer_norm"] + ":" + channels["device_id_norm"]

    device_medians = channels.groupby("pooled_device_id")["impedance_kohm"].median()
    pooled = devices.copy()
    pooled["device_median_impedance_kohm"] = pooled["pooled_device_id"].map(device_medians)
    pooled["median_in_80_200k_candidate_range"] = pooled[
        "device_median_impedance_kohm"
    ].between(DEVICE_MEDIAN_MIN_KOHM, DEVICE_MEDIAN_MAX_KOHM, inclusive="both")
    pooled["device_good_lower_kohm"] = (
        pooled["device_median_impedance_kohm"] * DEVICE_GOOD_LOWER_MULTIPLIER
    )
    pooled["device_good_upper_kohm"] = (
        pooled["device_median_impedance_kohm"] * DEVICE_GOOD_UPPER_MULTIPLIER
    )
    thresholds = pooled.set_index("pooled_device_id")[[
        "device_median_impedance_kohm",
        "device_good_lower_kohm",
        "device_good_upper_kohm",
    ]]
    channels = channels.join(thresholds, on="pooled_device_id", validate="many_to_one")
    channels["device_specific_class"] = np.select(
        [
            channels["impedance_kohm"] < channels["device_good_lower_kohm"],
            channels["impedance_kohm"] > channels["device_good_upper_kohm"],
        ],
        ["below", "above"],
        default="good",
    )
    counts = channels.groupby("pooled_device_id")["device_specific_class"].agg(
        n_measured_channels="size",
        n_device_good=lambda x: int((x == "good").sum()),
        n_device_below=lambda x: int((x == "below").sum()),
        n_device_above=lambda x: int((x == "above").sum()),
    )
    pooled = pooled.merge(counts, on="pooled_device_id", how="left", validate="one_to_one")
    pooled["device_specific_good_yield_percent"] = (
        100 * pooled["n_device_good"] / pooled["yield_denominator"]
    )
    eligible = pooled[
        pooled["median_in_80_200k_candidate_range"]
        & (pooled["device_specific_good_yield_percent"] >= MIN_DEVICE_YIELD)
    ].copy()
    selected = eligible.sort_values(
        ["device_specific_good_yield_percent", "device_median_impedance_kohm", "pooled_device_id"],
        ascending=[False, True, True],
    ).head(N_BEST_YIELD_DEVICES).copy()
    selected = selected.sort_values(
        ["device_median_impedance_kohm", "pooled_device_id"]
    ).reset_index(drop=True)
    selected["display_device_index"] = np.arange(1, len(selected) + 1)
    selected_channels = channels[
        channels["pooled_device_id"].isin(selected["pooled_device_id"])
    ].merge(
        selected[["pooled_device_id", "display_device_index"]],
        on="pooled_device_id",
        how="left",
        validate="many_to_one",
    )
    selected_channels["display_class"] = np.select(
        [
            selected_channels["impedance_kohm"] < DISPLAY_MIN_KOHM,
            selected_channels["impedance_kohm"] > DISPLAY_MAX_KOHM,
        ],
        ["below_display", "above_display"],
        default="visible",
    )

    box_rows = []
    for device_id, group in selected_channels.groupby("pooled_device_id", sort=False):
        values = group["impedance_kohm"].to_numpy(float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        box_rows.append(
            {
                "pooled_device_id": device_id,
                "display_device_index": int(group["display_device_index"].iloc[0]),
                "n_channel_points": len(values),
                "q1_impedance_kohm": q1,
                "median_impedance_kohm": median,
                "q3_impedance_kohm": q3,
                "iqr_impedance_kohm": iqr,
                "lower_whisker_observed_kohm": float(np.min(values[values >= lower_fence])),
                "upper_whisker_observed_kohm": float(np.max(values[values <= upper_fence])),
                "n_below_display": int(np.sum(values < DISPLAY_MIN_KOHM)),
                "n_above_display": int(np.sum(values > DISPLAY_MAX_KOHM)),
            }
        )

    selected.to_csv(DATA / "SF04b_pooled_selected_devices.csv", index=False)
    selected_channels.to_csv(DATA / "SF04b_pooled_selected_channel_points.csv", index=False)
    pd.DataFrame(box_rows).sort_values("display_device_index").to_csv(
        DATA / "SF04b_pooled_boxplot_statistics.csv", index=False
    )
    pooled.to_csv(DATA / "SF04b_all_source_devices_candidate_and_yield_audit.csv", index=False)

    positions = selected["display_device_index"].to_numpy(int)
    values_by_device = [
        selected_channels.loc[
            selected_channels["display_device_index"] == position, "impedance_kohm"
        ].to_numpy(float)
        for position in positions
    ]
    yields = selected["device_specific_good_yield_percent"].to_numpy(float)

    fig, (ax_imp, ax_yield) = plt.subplots(
        2,
        1,
        figsize=(7.2, 3.7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )
    threshold_color = "#8a9892"
    threshold_lowers = selected["device_good_lower_kohm"].to_numpy(float)
    threshold_uppers = selected["device_good_upper_kohm"].to_numpy(float)
    ax_imp.vlines(
        positions,
        threshold_lowers,
        threshold_uppers,
        color=threshold_color,
        linewidth=0.45,
        alpha=0.22,
        zorder=1,
    )
    cap_half_width = 0.20
    for position, lower, upper in zip(positions, threshold_lowers, threshold_uppers):
        ax_imp.hlines(
            [lower, upper],
            position - cap_half_width,
            position + cap_half_width,
            color=threshold_color,
            linewidth=0.45,
            alpha=0.22,
            zorder=1,
        )
    ax_imp.boxplot(
        values_by_device,
        positions=positions,
        widths=0.58,
        whis=1.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#202020", "linewidth": 0.65},
        boxprops={"facecolor": "none", "edgecolor": "#3f5f58", "linewidth": 0.45},
        whiskerprops={"color": "#3f5f58", "linewidth": 0.4},
        capprops={"color": "#3f5f58", "linewidth": 0.4},
        zorder=5,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    for position, values in zip(positions, values_by_device):
        jitter = rng.uniform(-0.24, 0.24, len(values))
        visible = (values >= DISPLAY_MIN_KOHM) & (values <= DISPLAY_MAX_KOHM)
        ax_imp.scatter(
            position + jitter[visible],
            values[visible],
            s=3.2,
            color="#40635b",
            alpha=0.48,
            linewidths=0,
            rasterized=False,
            zorder=4,
        )
        below = values < DISPLAY_MIN_KOHM
        above = values > DISPLAY_MAX_KOHM
        ax_imp.scatter(
            position + jitter[below],
            np.full(int(below.sum()), DISPLAY_MIN_KOHM * 1.025),
            marker="v",
            s=2.8,
            color="#9b4b47",
            alpha=0.52,
            linewidths=0,
            rasterized=False,
            zorder=4,
        )
        ax_imp.scatter(
            position + jitter[above],
            np.full(int(above.sum()), DISPLAY_MAX_KOHM / 1.025),
            marker="^",
            s=2.8,
            color="#526b91",
            alpha=0.52,
            linewidths=0,
            rasterized=False,
            zorder=4,
        )
    ax_imp.set_yscale("log")
    ax_imp.set_ylim(DISPLAY_MIN_KOHM, DISPLAY_MAX_KOHM)
    ax_imp.set_ylabel("Impedance at 1 kHz (kΩ)")
    ax_imp.spines[["top", "right"]].set_visible(False)
    ax_imp.tick_params(width=0.6, length=2.5)

    ax_yield.bar(positions, yields, width=0.72, color="#7186aa", linewidth=0)
    ax_yield.set_ylim(MIN_DEVICE_YIELD - 4, 101)
    ax_yield.set_yticks([75, 80, 90, 100])
    ax_yield.set_ylabel("Yield (%)")
    n_devices = len(selected)
    ticks = np.unique(np.r_[1, np.arange(10, n_devices + 1, 10), n_devices]).astype(int)
    ax_yield.set_xticks(ticks, [str(value) for value in ticks])
    ax_yield.set_xlim(0.25, n_devices + 0.75)
    ax_yield.set_xlabel("Pooled device index (ordered by median impedance)")
    ax_yield.spines[["top", "right"]].set_visible(False)
    ax_yield.tick_params(width=0.6, length=2.5)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.13, top=0.98)
    save(fig, "SF04b_pooled_W62_W71_W72_device_median_good_devices")

    per_wafer = selected.groupby("wafer_norm").agg(
        n_selected_devices=("pooled_device_id", "size"),
        median_device_specific_yield_percent=("device_specific_good_yield_percent", "median"),
        q1_device_specific_yield_percent=("device_specific_good_yield_percent", lambda x: x.quantile(0.25)),
        q3_device_specific_yield_percent=("device_specific_good_yield_percent", lambda x: x.quantile(0.75)),
    ).reset_index()
    per_wafer.to_csv(DATA / "SF04b_pooled_selected_device_stats_by_source_wafer.csv", index=False)
    denominator_reconciliation = (
        selected.groupby(["yield_denominator", "refs_enabled", "n_refs_excluded"])
        .agg(
            n_devices=("pooled_device_id", "size"),
            n_measured_nonreference_channels=("n_measured_channels", "sum"),
            n_missing_analyzable_channels=("n_missing_channels", "sum"),
            n_good_channels=("n_device_good", "sum"),
            n_below_device_range=("n_device_below", "sum"),
            n_above_device_range=("n_device_above", "sum"),
        )
        .reset_index()
    )
    denominator_reconciliation["n_nominal_physical_contacts"] = (
        128 * denominator_reconciliation["n_devices"]
    )
    denominator_reconciliation["n_excluded_reference_contacts"] = (
        denominator_reconciliation["n_refs_excluded"]
        * denominator_reconciliation["n_devices"]
    )
    denominator_reconciliation["total_analyzable_denominator"] = (
        denominator_reconciliation["yield_denominator"]
        * denominator_reconciliation["n_devices"]
    )
    denominator_reconciliation.to_csv(
        DATA / "SF04b_selected_device_denominator_reconciliation.csv", index=False
    )

    n_selected_110 = int((selected["yield_denominator"] == 110).sum())
    n_selected_128 = int((selected["yield_denominator"] == 128).sum())
    n_excluded_refs = int(selected["n_refs_excluded"].sum())
    n_missing_analyzable = int(selected["n_missing_channels"].sum())
    n_good_channels = int(selected["n_device_good"].sum())
    total_analyzable = int(selected["yield_denominator"].sum())
    pooled_channel_yield = 100 * n_good_channels / total_analyzable
    overall_device_eligibility = 100 * len(eligible) / len(pooled)
    conditional_device_eligibility = (
        100 * len(eligible) / pooled["median_in_80_200k_candidate_range"].sum()
    )
    summary = {
        "source_wafers": [f"W{x}" for x in WAFERS],
        "n_source_devices": int(len(devices)),
        "n_median_range_candidate_devices": int(pooled["median_in_80_200k_candidate_range"].sum()),
        "n_devices_meeting_candidate_and_75_percent_yield_criteria": int(len(eligible)),
        "n_selected_devices": int(len(selected)),
        "n_selected_channel_points": int(len(selected_channels)),
        "n_selected_devices_with_128_channel_denominator": n_selected_128,
        "n_selected_devices_with_110_channel_denominator": n_selected_110,
        "n_nominal_physical_contacts_selected": int(128 * len(selected)),
        "n_excluded_local_reference_contacts_selected": n_excluded_refs,
        "n_analyzable_channel_denominator_selected": total_analyzable,
        "n_missing_analyzable_channel_measurements_selected": n_missing_analyzable,
        "n_good_channels_selected": n_good_channels,
        "n_below_device_range_selected": int(selected["n_device_below"].sum()),
        "n_above_device_range_selected": int(selected["n_device_above"].sum()),
        "pooled_channel_yield_percent": pooled_channel_yield,
        "overall_device_eligibility_rate_percent": overall_device_eligibility,
        "conditional_candidate_eligibility_rate_percent": conditional_device_eligibility,
        "top100_selection_rate_among_all_devices_percent": 100 * len(selected) / len(pooled),
        "top100_selection_rate_among_eligible_devices_percent": 100 * len(selected) / len(eligible),
        "candidate_device_median_range_kohm": [DEVICE_MEDIAN_MIN_KOHM, DEVICE_MEDIAN_MAX_KOHM],
        "device_good_channel_rule": "0.7-1.5 times each device median impedance",
        "selection_rule": f"top {N_BEST_YIELD_DEVICES} device-specific yields among devices with candidate median in 80-200 kOhm and good-channel yield >= {MIN_DEVICE_YIELD:.0f}%",
        "selected_yield_median_percent": float(np.median(yields)),
        "selected_yield_q1_percent": float(np.quantile(yields, 0.25)),
        "selected_yield_q3_percent": float(np.quantile(yields, 0.75)),
        "device_order": "ascending device median impedance; source wafer not displayed",
        "display_impedance_range_kohm": [DISPLAY_MIN_KOHM, DISPLAY_MAX_KOHM],
        "threshold_marks": "light capped vertical marks show each device's 0.7x-1.5x median range",
        "vector_points": True,
    }
    (RUN / "RUN_MANIFEST.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pass  # Source tables and graphics are written above.
    print(json.dumps(summary, indent=2))
    print(f"Saved pooled SF4b run to {RUN}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Supplementary Figure 4A from wafer measurements")
    parser.add_argument("--out", type=Path, required=True)
    RUN = parser.parse_args().out
    FIGURES = RUN / "figures"
    DATA = RUN / "plotting_data"
    main()
