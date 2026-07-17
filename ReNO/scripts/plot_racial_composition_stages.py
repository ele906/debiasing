"""
Composite (stacked) bar chart of racial composition at key optimization
stages -- 0th, 10th, 20th, 30th, 40th, 50th iteration, and "best" -- read
from a composition.csv produced by scripts/analyze_racial_composition.ipynb
(columns: prompt, stage, race, count, pct_of_faces; stage in
{'init','0'..'49','best'}).

Writes one stacked bar chart per prompt (plots/<prompt>_stages.png) plus a
combined chart with all prompts side by side (plots/all_prompts_stages.png).

Usage:
    python scripts/plot_racial_composition_stages.py \\
        --csv outputs/racial_composition_baseline/composition.csv \\
        --out_dir outputs/racial_composition_baseline/plots
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

# dataviz reference palette, categorical slots 1-7 (fixed order == CVD safety)
RACE_COLORS = {
    "White": "#2a78d6",  # blue
    "Black": "#008300",  # green
    "Latino_Hispanic": "#e87ba4",  # magenta
    "East Asian": "#eda100",  # yellow
    "Southeast Asian": "#1baf7a",  # aqua
    "Indian": "#eb6834",  # orange
    "Middle Eastern": "#4a3aa7",  # violet
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

STAGE_LABELS = ["0", "10", "20", "30", "40", "50", "best"]
SEGMENT_GAP = 0.008  # fraction-of-total-height gap between stacked segments


def resolve_stage(stages_available, label):
    if label == "best":
        return "best" if "best" in stages_available else None
    if label in stages_available:
        return label
    numeric_stages = sorted((s for s in stages_available if s.isdigit()), key=int)
    return numeric_stages[-1] if numeric_stages else None


def build_pivot(df, prompt_slug):
    prompt_rows = df[df["prompt"] == prompt_slug]
    stages_available = set(prompt_rows["stage"])
    rows = []
    for label in STAGE_LABELS:
        actual_stage = resolve_stage(stages_available, label)
        if actual_stage is None:
            continue
        stage_rows = prompt_rows[prompt_rows["stage"] == actual_stage]
        for _, r in stage_rows.iterrows():
            rows.append(
                {"stage": label, "race": r["race"], "pct_of_faces": r["pct_of_faces"]}
            )
    pivot = pd.DataFrame(rows).pivot_table(
        index="stage", columns="race", values="pct_of_faces", fill_value=0.0
    )
    pivot = pivot.reindex([s for s in STAGE_LABELS if s in pivot.index])
    pivot = pivot.reindex(columns=RACE_LABELS, fill_value=0.0)
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
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_stacked(ax, pivot, x_positions, bar_width, add_labels=True):
    """Draw one stacked-bar group with a thin surface gap between segments."""
    bottoms = [0.0] * len(pivot)
    for race in RACE_LABELS:
        heights = pivot[race].to_numpy()
        gapped_heights = [max(h - SEGMENT_GAP, 0.0) if h > 0 else 0.0 for h in heights]
        ax.bar(
            x_positions,
            gapped_heights,
            width=bar_width,
            bottom=bottoms,
            color=RACE_COLORS[race],
            label=race if add_labels else None,
            zorder=2,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]


def add_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
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
    return legend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"stage": str})
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = sorted(df["prompt"].unique())

    # per-prompt composite bar charts
    for prompt_slug in prompts:
        pivot = build_pivot(df, prompt_slug)
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
        x = range(len(pivot))
        plot_stacked(ax, pivot, x, bar_width=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(pivot.index, color=INK_MUTED)
        ax.set_xlabel("Optimization stage", color=INK_SECONDARY, fontsize=9)
        style_axes(ax, prompt_slug.replace("_", " "))
        add_legend(fig, ax)
        fig.tight_layout(rect=(0, 0, 0.82, 1))
        out_path = os.path.join(args.out_dir, f"{prompt_slug}_stages.png")
        fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # combined chart: all prompts, grouped by stage, side by side
    n_stages = len(STAGE_LABELS)
    group_width = 0.85
    bar_width = group_width / len(prompts)
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    for i, prompt_slug in enumerate(prompts):
        pivot = build_pivot(df, prompt_slug)
        offset = (i - (len(prompts) - 1) / 2) * bar_width
        x = [s + offset for s in range(len(pivot))]
        plot_stacked(ax, pivot, x, bar_width=bar_width * 0.9, add_labels=(i == 0))

    ax.set_xticks(range(n_stages))
    ax.set_xticklabels(STAGE_LABELS, color=INK_MUTED)
    ax.set_xlabel("Optimization stage", color=INK_SECONDARY, fontsize=9)
    style_axes(ax, "Racial composition by prompt & generation stage")

    add_legend(fig, ax)
    # secondary legend for prompts via a text caption (avoids a second color channel)
    caption = "  |  ".join(
        f"group {i + 1}: {p.replace('_', ' ')}" for i, p in enumerate(prompts)
    )
    fig.text(0.02, 0.005, caption, color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    out_path = os.path.join(args.out_dir, "final_composition.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()


