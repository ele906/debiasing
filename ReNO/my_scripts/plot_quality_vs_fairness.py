"""
Plots per-iteration quality reward (ImageReward/HPS/PickScore/CLIP) against
the Fairness KL loss, to show how much visual quality is traded away as
optimization pushes the race/gender distribution toward the target.

Reads rewards.csv files written by training/trainer.py (one per
{save_dir}/{prompt_slug}/seed{N}/rewards.csv) -- only present for runs
with --enable_fairness (Fairness must be one of the logged columns).

Usage:
    python scripts/plot_quality_vs_fairness.py \\
        --data_dir outputs/two_person_fairness \\
        --out_dir outputs/two_person_fairness/plots
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
QUALITY_COLOR = "#2a78d6"
FAIRNESS_COLOR = "#eb6834"


def style_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=14, loc="left")
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Optimization iteration", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_runs(data_dir):
    rows = []
    for rewards_csv in sorted(glob.glob(os.path.join(data_dir, "*", "seed*", "rewards.csv"))):
        seed_dir = os.path.dirname(rewards_csv)
        prompt_slug = os.path.basename(os.path.dirname(seed_dir))
        seed = os.path.basename(seed_dir)
        df = pd.read_csv(rewards_csv)
        df["prompt"] = prompt_slug
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise SystemExit(f"no rewards.csv files found under {data_dir}")
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--quality_metric",
        type=str,
        default="ImageReward",
        choices=["ImageReward", "HPS", "PickScore", "CLIP", "total"],
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_runs(args.data_dir)
    if "Fairness" not in df.columns:
        raise SystemExit(
            "no 'Fairness' column found -- rerun with --enable_fairness so "
            "the fairness KL loss is logged alongside quality rewards"
        )

    prompts = sorted(df["prompt"].unique())
    for prompt_slug in prompts:
        prompt_df = df[df["prompt"] == prompt_slug]
        quality_by_iter = prompt_df.groupby("iteration")[args.quality_metric].mean()
        fairness_by_iter = prompt_df.groupby("iteration")["Fairness"].mean()

        fig, ax1 = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
        ax1.plot(
            quality_by_iter.index,
            quality_by_iter.values,
            color=QUALITY_COLOR,
            linewidth=2,
            label=args.quality_metric,
        )
        style_axes(ax1, prompt_slug.replace("_", " "), f"{args.quality_metric} (mean across seeds)")
        ax1.yaxis.label.set_color(QUALITY_COLOR)
        ax1.tick_params(axis="y", colors=QUALITY_COLOR)

        ax2 = ax1.twinx()
        ax2.plot(
            fairness_by_iter.index,
            fairness_by_iter.values,
            color=FAIRNESS_COLOR,
            linewidth=2,
            linestyle="--",
            label="Fairness KL",
        )
        ax2.set_ylabel("Fairness KL loss (mean across seeds)", color=FAIRNESS_COLOR, fontsize=9)
        ax2.tick_params(axis="y", colors=FAIRNESS_COLOR, labelsize=9, length=0)
        for spine in ("top", "left"):
            ax2.spines[spine].set_visible(False)
        ax2.spines["right"].set_color(GRIDLINE)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
            fontsize=8.5,
            labelcolor=INK_SECONDARY,
        )

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{prompt_slug}_quality_vs_fairness.png")
        fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
