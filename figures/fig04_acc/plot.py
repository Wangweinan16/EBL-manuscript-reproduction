from __future__ import annotations


import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "reproduced_acc_panels"

SESSION_CONFIG = (
    ("20230413", "Day-115", np.arange(206, 235)),
    ("20230414", "Day-116", np.arange(243, 275)),
    ("20230418", "Day-120", np.arange(293, 323)),
    ("20230617", "Day-180", np.arange(124, 156)),
)

EVENT_INDEX_MATLAB = 3
WINDOW_S = (-2.0, 2.0)
LOW_RATE_THRESHOLD_HZ = 0.25
DISPLAY_LIMITS = (-5.0, 0.0)

HEATMAP_SIZE_IN = (4.7, 1.8)
PANEL2_SIZE_IN = (2.05, 1.55)
PANEL3_SIZE_IN = (2.15, 1.74)
PANEL4_SIZE_IN = (7.2, 2.45)
PANEL4_OWN_SIZE_IN = (7.2, 2.05)
PANEL4_COMPACT_SIZE_IN = (1.7, 1.9)
PANEL4_2X2_ORIGINALSIZE_IN = (3.45, 3.55)

TICK_FONT_PT = 6
LEGEND_FONT_PT = 6
AXIS_LABEL_FONT_PT = 7
TITLE_FONT_PT = 8
PLOT_BOX_STROKE_PT = 0.2

LEFT_ACTION_COLOR = np.array([1.0, 0.85, 0.0])
RIGHT_ACTION_COLOR = np.array([0.4, 0.8, 0.2])
STEP_COLORS = {
    1: np.array([0.8510, 0.6588, 0.1804]),
    2: np.array([0.5302, 0.3827, 0.0470]),
    3: np.array([0.3686, 0.6196, 0.2627]),
}

LEFT_CURVE_COLORS = np.array(
    [
        [0.9843, 0.9216, 0.6275],
        [0.9412, 0.8157, 0.3765],
        [0.8510, 0.6588, 0.1804],
        [0.6627, 0.4784, 0.0588],
    ]
)
RIGHT_CURVE_COLORS = np.array(
    [
        [0.8039, 0.9098, 0.6902],
        [0.5961, 0.8000, 0.4745],
        [0.3686, 0.6196, 0.2627],
        [0.2078, 0.4196, 0.1490],
    ]
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": TICK_FONT_PT,
            "axes.labelsize": AXIS_LABEL_FONT_PT,
            "axes.titlesize": TITLE_FONT_PT,
            "axes.linewidth": PLOT_BOX_STROKE_PT,
            "xtick.labelsize": TICK_FONT_PT,
            "ytick.labelsize": TICK_FONT_PT,
            "xtick.major.width": PLOT_BOX_STROKE_PT,
            "ytick.major.width": PLOT_BOX_STROKE_PT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    fig.savefig(base.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def plot_heatmap(data: dict[str, object]) -> plt.Figure:
    ordered = np.asarray(data["z_scored_rastermap_ordered"])
    choices = np.asarray(data["choices"])
    bins_per_trial = int(np.asarray(data["bins_per_trial"])[0])
    n_units, n_columns = ordered.shape

    fig, ax = plt.subplots(figsize=HEATMAP_SIZE_IN, dpi=150)
    ax.imshow(
        -ordered,
        cmap="gray",
        vmin=DISPLAY_LIMITS[0],
        vmax=DISPLAY_LIMITS[1],
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        extent=(0, n_columns, 0.5, n_units + 0.5),
    )

    for trial_index, choice in enumerate(choices):
        color = LEFT_ACTION_COLOR if choice == 1 else RIGHT_ACTION_COLOR
        ax.add_patch(
            Rectangle(
                (trial_index * bins_per_trial, 0),
                bins_per_trial,
                n_units + 1,
                facecolor=color,
                edgecolor="none",
                alpha=0.05,
            )
        )
    for x_value in range(bins_per_trial, n_columns, bins_per_trial):
        ax.axvline(x_value, color=(0.3, 0.3, 0.3), linewidth=0.5)

    legend_x = n_columns + 0.25 * bins_per_trial
    legend_width = 0.5 * bins_per_trial
    for y_value, color in (
        (n_units - 1.0, LEFT_ACTION_COLOR),
        (n_units - 9.0, RIGHT_ACTION_COLOR),
    ):
        ax.add_patch(
            Rectangle(
                (legend_x, y_value),
                legend_width,
                3,
                facecolor=color,
                edgecolor="none",
                alpha=0.6,
            )
        )
    ax.text(
        n_columns + 1.1 * bins_per_trial,
        n_units + 0.5,
        "Left actions",
        fontsize=LEGEND_FONT_PT,
        va="center",
    )
    ax.text(
        n_columns + 1.1 * bins_per_trial,
        n_units - 7.5,
        "Right actions",
        fontsize=LEGEND_FONT_PT,
        va="center",
    )

    example_row = data["example_unit_row_zero_based"]
    if str(data["date"]) == "04182023" and example_row is not None:
        ax.scatter(
            n_columns + 0.38 * bins_per_trial,
            int(example_row) + 1,
            marker="<",
            s=24,
            facecolor="black",
            edgecolor="black",
            linewidth=0,
            clip_on=False,
            zorder=10,
        )

    ax.set_xlim(0, n_columns + bins_per_trial)
    ax.set_ylim(-2, n_units + 3)
    ax.set_yticks([1, n_units])
    ax.set_yticklabels(["1", str(n_units)])
    ax.set_ylabel("Units (RasterMap re-ordered)")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, bins_per_trial / 2, bins_per_trial])
    ax.set_xticklabels(["-2", "0", "2"])
    for label, ha in zip(ax.get_xticklabels(), ("right", "center", "left")):
        label.set_horizontalalignment(ha)
    ax.tick_params(axis="x", bottom=True, labelbottom=True, length=2, pad=1)
    ax.set_xlabel("Action onset (s)", labelpad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.20, top=0.96)
    return fig


def plot_panel2() -> tuple[plt.Figure, dict[str, np.ndarray]]:
    data = loadmat(HERE / "panel2_example_curve_data.mat", simplify_cells=True)
    mean_all = data["mean_all"]
    sem_all = data["sem_all"]
    x = np.arange(-2.0, 2.0, 0.01)

    fig, ax = plt.subplots(figsize=PANEL2_SIZE_IN, dpi=150)
    color_groups = (LEFT_CURVE_COLORS, RIGHT_CURVE_COLORS)
    labels = ("Left", "Right")

    for action_index in (1, 0):
        for session_index in range(4):
            mean = np.asarray(mean_all[action_index, session_index]).ravel()
            sem = np.asarray(sem_all[action_index, session_index]).ravel()
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=color_groups[action_index][session_index],
                alpha=0.20,
                linewidth=0,
            )

    days = (115, 116, 120, 180)
    for action_index in range(2):
        for session_index, day in enumerate(days):
            ax.plot(
                x,
                np.asarray(mean_all[action_index, session_index]).ravel(),
                color=color_groups[action_index][session_index],
                linewidth=1.0,
                label=f"{labels[action_index]} (day-{day})",
            )

    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 50)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([0, 50])
    ax.set_xlabel("Time from action onset (sec)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)
    legend = ax.legend(
        loc="upper left",
        fontsize=LEGEND_FONT_PT,
        frameon=False,
        fancybox=False,
        framealpha=1,
        edgecolor=(0.25, 0.25, 0.25),
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=1.7,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.98)
    return fig, {"mean_all": mean_all, "sem_all": sem_all, "time_s": x}


def decoding_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "hanqing_blue_red", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], N=256
    )


def panel3_line_color(mean_score: float) -> tuple[float, float, float, float]:
    index = int(np.clip(np.ceil((mean_score - 0.4) / 0.55 * 256), 1, 256))
    return decoding_colormap()((index - 1) / 255)


def plot_panel3() -> tuple[plt.Figure, dict[str, np.ndarray]]:
    data = loadmat(
        HERE / "panel3_decoding_score_distribution.mat", simplify_cells=True
    )
    scores = np.asarray(data["scores"], dtype=float)
    scores_sf = np.asarray(data["scores_sf"], dtype=float)
    mean_scores = scores.mean(axis=1)
    order = np.argsort(mean_scores)

    fig, ax = plt.subplots(figsize=PANEL3_SIZE_IN, dpi=150)
    for unit_index in order:
        color = panel3_line_color(float(mean_scores[unit_index]))
        ax.plot(
            np.arange(1, 5),
            scores[unit_index],
            marker="s",
            markersize=3.1,
            linewidth=1.0,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )

    correlations = np.array(
        [np.corrcoef(scores[:, i], scores[:, i + 1])[0, 1] for i in range(3)]
    )
    for i, correlation in enumerate(correlations):
        ax.text(
            i + 1.5,
            0.95,
            f"r={correlation:.2f}",
            ha="center",
            va="center",
            fontsize=TICK_FONT_PT,
        )

    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["day-115", "day-116", "day-120", "day-180"])
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel("Left vs. Right decoding score")
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False)

    scalar = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0.5, vmax=0.95), cmap=decoding_colormap()
    )
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.025, fraction=0.065)
    colorbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9])
    colorbar.set_label("Session-averaged decoding score", fontsize=AXIS_LABEL_FONT_PT)
    colorbar.ax.tick_params(labelsize=TICK_FONT_PT)
    # Hanqing's July revision intentionally omits the chance-level line.
    fig.subplots_adjust(left=0.18, right=0.84, bottom=0.20, top=0.98)
    return fig, {
        "scores": scores,
        "scores_shuffled": scores_sf,
        "mean_scores": mean_scores,
        "adjacent_session_correlations": correlations,
    }


def orient_components(
    scores: np.ndarray, labels: np.ndarray, components: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    oriented_scores = scores.copy()
    oriented_components = components.copy()

    class1_x = oriented_scores[labels == 1, 0].mean()
    class2_x = oriented_scores[labels == 2, 0].mean()
    if class1_x > class2_x:
        oriented_scores[:, 0] *= -1
        oriented_components[:, 0] *= -1

    class1_y = oriented_scores[labels == 1, 1].mean()
    class2_y = oriented_scores[labels == 2, 1].mean()
    if class2_y < class1_y:
        oriented_scores[:, 1] *= -1
        oriented_components[:, 1] *= -1
    return oriented_scores, oriented_components


def fit_lda(scores: np.ndarray, labels: np.ndarray) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="svd").fit(scores[:, :2], labels)


def accuracy(
    model: LinearDiscriminantAnalysis, scores: np.ndarray, labels: np.ndarray
) -> float:
    return float(np.mean(model.predict(scores[:, :2]) == labels))


def plot_lda_boundaries(
    ax: plt.Axes,
    model: LinearDiscriminantAnalysis,
    shift: np.ndarray | None = None,
    linestyle: str = ":",
    alpha: float = 0.18,
    linewidth: float = 1.5,
    color: tuple[float, float, float] = (0.35, 0.35, 0.35),
) -> None:
    shift = np.zeros(2) if shift is None else np.asarray(shift)
    solution_x = 1.4
    for i in range(3):
        for j in range(i + 1, 3):
            linear = model.coef_[i] - model.coef_[j]
            constant = model.intercept_[i] - model.intercept_[j]
            x = (
                np.array([-50.0, solution_x])
                if i == 0
                else np.array([solution_x, 50.0])
            )
            y = -(linear[0] * x + constant) / linear[1]
            ax.plot(
                x + shift[0],
                y + shift[1],
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                solid_capstyle="butt",
                dash_capstyle="butt",
                zorder=1,
            )


def scatter_sequence_steps(ax: plt.Axes, scores: np.ndarray, labels: np.ndarray) -> None:
    ax.scatter(
        scores[labels == 3, 0],
        scores[labels == 3, 1],
        s=20,
        marker="+",
        color=STEP_COLORS[3],
        linewidth=1.1,
        zorder=3,
    )
    ax.scatter(
        scores[labels == 2, 0],
        scores[labels == 2, 1],
        s=20,
        marker="x",
        color=STEP_COLORS[2],
        linewidth=1.1,
        zorder=3,
    )
    ax.scatter(
        scores[labels == 1, 0],
        scores[labels == 1, 1],
        s=13,
        marker="d",
        facecolors="none",
        edgecolors=STEP_COLORS[1],
        linewidth=1.0,
        zorder=3,
    )


def scatter_sequence_steps_compact(
    ax: plt.Axes, scores: np.ndarray, labels: np.ndarray
) -> None:
    ax.scatter(
        scores[labels == 3, 0],
        scores[labels == 3, 1],
        s=5.5,
        marker="+",
        color=STEP_COLORS[3],
        linewidth=0.48,
        zorder=3,
    )
    ax.scatter(
        scores[labels == 2, 0],
        scores[labels == 2, 1],
        s=5.5,
        marker="x",
        color=STEP_COLORS[2],
        linewidth=0.48,
        zorder=3,
    )
    ax.scatter(
        scores[labels == 1, 0],
        scores[labels == 1, 1],
        s=4.8,
        marker="d",
        facecolors="none",
        edgecolors=STEP_COLORS[1],
        linewidth=0.45,
        zorder=3,
    )


def style_population_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_xticks([-50, -25, 25, 50], minor=True)
    ax.set_yticks([-50, -25, 25, 50], minor=True)
    ax.grid(which="major", color=(0.55, 0.55, 0.55), alpha=0.35, linewidth=0.45)
    ax.grid(which="minor", color=(0.55, 0.55, 0.55), alpha=0.25, linewidth=0.4)
    ax.tick_params(top=True, right=True, labeltop=False, labelright=False, length=2)
    ax.tick_params(which="minor", length=0)


def style_compact_population_axis(
    ax: plt.Axes, row: int, column: int, day_label: str
) -> None:
    style_population_axis(ax)
    ax.grid(False, which="both")
    ax.tick_params(length=0, width=PLOT_BOX_STROKE_PT, pad=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(
        day_label.replace("Day-", "Day "),
        fontsize=TITLE_FONT_PT,
        fontweight="normal",
        pad=0.8,
    )
    if row == 1:
        ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FONT_PT, labelpad=0.4)
    else:
        ax.set_xlabel("")
    if column == 0:
        ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FONT_PT, labelpad=0.5)
    else:
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_BOX_STROKE_PT)


def style_population_axis_2x2(ax: plt.Axes, row: int, column: int, day_label: str) -> None:
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_title(
        day_label.replace("Day-", "Day "),
        fontsize=TITLE_FONT_PT,
        fontweight="normal",
        pad=2.0,
    )
    ax.set_xlabel("PC1" if row == 1 else "", fontsize=AXIS_LABEL_FONT_PT, labelpad=1.0)
    ax.set_ylabel("PC2" if column == 0 else "", fontsize=AXIS_LABEL_FONT_PT, labelpad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_BOX_STROKE_PT)


def fixed_day1_population_analysis(
    population: list[dict[str, np.ndarray]],
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    first = population[0]
    pca = PCA(svd_solver="full").fit(first["firing"])
    first_scores = pca.transform(first["firing"])
    first_scores, components = orient_components(
        first_scores, first["labels"], pca.components_.T
    )
    center = pca.mean_
    model = fit_lda(first_scores, first["labels"])
    first_mean = first_scores[:, :2].mean(axis=0)

    fig, axes = plt.subplots(1, 4, figsize=PANEL4_SIZE_IN, dpi=150)
    all_scores = []
    all_labels = []
    explained = []
    score_table = []

    for session_index, ((_, day, _), session, ax) in enumerate(
        zip(SESSION_CONFIG, population, axes)
    ):
        scores = (session["firing"] - center) @ components
        session_mean = scores[:, :2].mean(axis=0)
        shift = session_mean - first_mean
        variances = scores.var(axis=0, ddof=1)
        ev = variances[:2] / variances.sum()

        pre = accuracy(model, scores, session["labels"])
        shifted = accuracy(model, scores - np.pad(shift, (0, scores.shape[1] - 2)), session["labels"])
        local_model = fit_lda(scores, session["labels"])
        local = accuracy(local_model, scores, session["labels"])

        plot_lda_boundaries(ax, model, linestyle=":", alpha=0.16, linewidth=1.5)
        if session_index > 0:
            plot_lda_boundaries(
                ax, model, shift=shift, linestyle="-", alpha=0.34, linewidth=1.7
            )
        scatter_sequence_steps(ax, scores, session["labels"])
        style_population_axis(ax)
        ax.set_xlabel(f"PC1 from d-115, EV={ev[0] * 100:.0f}%")
        ax.set_ylabel(f"PC2 from d-115, EV={ev[1] * 100:.0f}%")
        if session_index == 0:
            title = f"{day}: local\nDecoding score: {pre:.2f}"
        else:
            title = (
                f"{day}: stability test\n"
                f"Pre-shift: {pre:.2f}; Shifted: {shifted:.2f}\n"
                f"(Redo LDA: {local:.2f})"
            )
        ax.set_title(title, fontweight="bold", pad=4)

        all_scores.append(scores)
        all_labels.append(session["labels"])
        explained.append(ev)
        score_table.append([pre, shifted, local])

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.79, wspace=0.46)
    return fig, {
        "scores": np.asarray(all_scores, dtype=object),
        "labels": np.asarray(all_labels, dtype=object),
        "explained_variance_pc1_pc2": np.asarray(explained),
        "accuracy_pre_shift_shifted_redo": np.asarray(score_table),
        "lda_model": model,
        "day1_score_mean_pc1_pc2": first_mean,
        "day1_center": center,
        "day1_components": components,
    }


def plot_fixed_day1_population_compact(
    fixed: dict[str, np.ndarray],
) -> plt.Figure:
    scores_by_session = fixed["scores"]
    labels_by_session = fixed["labels"]
    model = fixed["lda_model"]
    first_mean = fixed["day1_score_mean_pc1_pc2"]

    fig, axes = plt.subplots(2, 2, figsize=PANEL4_COMPACT_SIZE_IN, dpi=300)
    for session_index, (ax, (_, day, _)) in enumerate(
        zip(axes.ravel(), SESSION_CONFIG)
    ):
        scores = np.asarray(scores_by_session[session_index])
        labels = np.asarray(labels_by_session[session_index])
        session_mean = scores[:, :2].mean(axis=0)
        shift = session_mean - first_mean

        plot_lda_boundaries(
            ax,
            model,
            linestyle=(0, (1.0, 1.0)),
            alpha=0.70,
            linewidth=0.50,
            color=(0.45, 0.45, 0.45),
        )
        if session_index > 0:
            plot_lda_boundaries(
                ax,
                model,
                shift=shift,
                linestyle="-",
                alpha=0.88,
                linewidth=0.58,
                color=(0.15, 0.15, 0.15),
            )
        scatter_sequence_steps_compact(ax, scores, labels)
        row, column = divmod(session_index, 2)
        style_compact_population_axis(ax, row, column, day)

    fig.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.08,
        hspace=0.03,
    )
    return fig


def plot_fixed_day1_population_2x2_originalsize(
    fixed: dict[str, np.ndarray],
) -> plt.Figure:
    scores_by_session = fixed["scores"]
    labels_by_session = fixed["labels"]
    model = fixed["lda_model"]
    first_mean = fixed["day1_score_mean_pc1_pc2"]

    fig, axes = plt.subplots(2, 2, figsize=PANEL4_2X2_ORIGINALSIZE_IN, dpi=300)
    for session_index, (ax, (_, day, _)) in enumerate(
        zip(axes.ravel(), SESSION_CONFIG)
    ):
        scores = np.asarray(scores_by_session[session_index])
        labels = np.asarray(labels_by_session[session_index])
        session_mean = scores[:, :2].mean(axis=0)
        shift = session_mean - first_mean

        plot_lda_boundaries(
            ax,
            model,
            linestyle=(0, (1.1, 1.1)),
            alpha=0.72,
            linewidth=0.85,
            color=(0.45, 0.45, 0.45),
        )
        if session_index > 0:
            plot_lda_boundaries(
                ax,
                model,
                shift=shift,
                linestyle="-",
                alpha=0.90,
                linewidth=0.95,
                color=(0.15, 0.15, 0.15),
            )
        scatter_sequence_steps(ax, scores, labels)
        row, column = divmod(session_index, 2)
        style_population_axis_2x2(ax, row, column, day)

    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.11,
        top=0.94,
        wspace=0.13,
        hspace=0.16,
    )
    return fig



def main():
    global OUTPUT_DIR
    cli = argparse.ArgumentParser(description="Render the four quantitative ACC panels.")
    cli.add_argument("--out", type=Path, required=True)
    OUTPUT_DIR = cli.parse_args().out
    configure_matplotlib()
    with np.load(HERE/"panel_inputs.npz") as data:
        heatmap = {key: data[key] for key in ["choices","bins_per_trial","example_unit_row_zero_based","z_scored_rastermap_ordered"]}
        heatmap["date"] = "04182023"
        population = [{"firing":data[f"firing_{i}"],"labels":data[f"labels_{i}"]} for i in range(4)]
        save_figure(plot_heatmap(heatmap),"Fig04b")
        for day in (115,116,180):
            values = {key:data[f"{key}_day{day}"] for key in ("choices","bins_per_trial","z_scored_rastermap_ordered")}
            values.update(date=f"day{day}",example_unit_row_zero_based=None)
            save_figure(plot_heatmap(values),f"FigS13_day{day}")
        save_figure(plot_panel2()[0],"Fig04c")
        save_figure(plot_panel3()[0],"Fig04d")
        diagnostic, fitted = fixed_day1_population_analysis(population)
        plt.close(diagnostic)
        save_figure(plot_fixed_day1_population_2x2_originalsize(fitted),"Fig04e",dpi=600)


if __name__ == "__main__":
    main()
