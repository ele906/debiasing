"""
Grouped bar chart comparing racial (and optionally gender) composition at
the "best" stage between a baseline run (no fairness reward) and a
debiased run (--enable_fairness), per prompt.

Reads two composition_by_stage.csv files produced by
scripts/racial_composition_analyze_by_stage.py (columns: prompt, stage,
race, count, pct_of_faces).

Usage:
    python scripts/plot_baseline_vs_debiased.py \\
        --baseline_csv outputs/two_person_check/composition_by_stage.csv \\
        --debiased_csv outputs/two_person_fairness/composition_by_stage.csv \\
        --out_dir outputs/two_person_fairness/plots
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]

RACE_COLORS = {
    "White": "#2a78d6",
    "Black": "#008300",
    "Latino_Hispanic": "#e87ba4",
    "East Asian": "#eda100",
    "Southeast Asian": "#1baf7a",
    "Indian": "#eb6834",
    "Middle Eastern": "#4a3aa7",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SEGMENT_GAP = 0.008


def best_stage_pivot(df, prompt_slug):
    prompt_rows = df[df["prompt"] == prompt_slug]
    stage = "best" if "best" in set(prompt_rows["stage"]) else prompt_rows["stage"].iloc[-1]
    rows = prompt_rows[prompt_rows["stage"] == stage]
    pivot = rows.set_index("race")["pct_of_faces"].reindex(RACE_LABELS, fill_value=0.0)
    return pivot


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=14, loc="left")
    ax.set_ylabel("Share of detected faces", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_LINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_stacked(ax, pivot, x, bar_width, add_labels):
    bottom = 0.0
    for race in RACE_LABELS:
        h = max(pivot[race] - SEGMENT_GAP, 0.0) if pivot[race] > 0 else 0.0
        ax.bar(
            x,
            h,
            width=bar_width,
            bottom=bottom,
            color=RACE_COLORS[race],
            label=race if add_labels else None,
            zorder=2,
        )
        bottom += pivot[race]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--debiased_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    baseline_df = pd.read_csv(args.baseline_csv, dtype={"stage": str})
    debiased_df = pd.read_csv(args.debiased_csv, dtype={"stage": str})

    prompts = sorted(set(baseline_df["prompt"]) & set(debiased_df["prompt"]))
    if not prompts:
        raise SystemExit("no shared prompt slugs between baseline_csv and debiased_csv")

    fig, ax = plt.subplots(figsize=(max(7.5, 3 * len(prompts)), 5.5), facecolor=SURFACE)
    bar_width = 0.32
    group_gap = 0.9

    for i, prompt_slug in enumerate(prompts):
        base_pivot = best_stage_pivot(baseline_df, prompt_slug)
        debias_pivot = best_stage_pivot(debiased_df, prompt_slug)
        x0 = i * group_gap - bar_width / 2 - 0.03
        x1 = i * group_gap + bar_width / 2 + 0.03
        plot_stacked(ax, base_pivot, x0, bar_width, add_labels=(i == 0))
        plot_stacked(ax, debias_pivot, x1, bar_width, add_labels=False)

    xticks, xlabels = [], []
    for i, prompt_slug in enumerate(prompts):
        xticks += [i * group_gap - bar_width / 2 - 0.03, i * group_gap + bar_width / 2 + 0.03]
        xlabels += ["baseline", "debiased"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, color=INK_MUTED, fontsize=8, rotation=0)

    # prompt group labels below the baseline/debiased tick labels
    for i, prompt_slug in enumerate(prompts):
        ax.text(
            i * group_gap,
            -0.12,
            prompt_slug.replace("_", " "),
            ha="center",
            va="top",
            color=INK_SECONDARY,
            fontsize=8.5,
            transform=ax.get_xaxis_transform(),
        )

    style_axes(ax, "Racial composition: baseline vs. debiased (best-image stage)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.0, 0.92),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        handlelength=1.2,
        handleheight=1.2,
    )

    fig.tight_layout(rect=(0, 0.05, 0.82, 1))
    out_path = os.path.join(args.out_dir, "baseline_vs_debiased.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
