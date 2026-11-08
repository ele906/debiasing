#!/usr/bin/env python3
"""Plot category-wise Feature 2023 changed-area trends from report CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Feature 2023 · landmark head-covering region")
    parser.add_argument("--matrix", action="store_true", help="render categories as rows and alpha values as columns")
    args = parser.parse_args()
    rows = [row for path in args.csv for row in csv.DictReader(path.open(newline=""))]
    groups = sorted({row["group"] for row in rows})
    if args.matrix:
        alphas = sorted({int(row["alpha"]) for row in rows})
        values = np.array([[next(float(row["mean_changed_area_pct_upper_roi"]) for row in rows if row["group"] == group and int(row["alpha"]) == alpha) for alpha in alphas] for group in groups])
        errors = np.array([[next(float(row["sd_changed_area_pct_upper_roi"]) for row in rows if row["group"] == group and int(row["alpha"]) == alpha) for alpha in alphas] for group in groups])
        fig, ax = plt.subplots(figsize=(13, max(5.5, len(groups) * .48)))
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(alphas)), [f"{alpha:+d}" for alpha in alphas])
        ax.set_yticks(range(len(groups)), [group.replace("_", " ").title() for group in groups])
        ax.set(xlabel="Intervention alpha", ylabel="Category", title=args.title)
        for i in range(len(groups)):
            for j in range(len(alphas)):
                ax.text(j, i, f"{values[i,j]:.1f}\n±{errors[i,j]:.1f}", ha="center", va="center", fontsize=7, color="white" if values[i,j] < values.max() * .62 else "black")
        fig.colorbar(image, ax=ax, label="Changed area within landmark region (%)")
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.get_cmap("tab20", max(len(groups), 1))
    for index, group in enumerate(groups):
        selected = sorted((row for row in rows if row["group"] == group), key=lambda row: int(row["alpha"]))
        x = np.array([int(row["alpha"]) for row in selected])
        y = np.array([float(row["mean_changed_area_pct_upper_roi"]) for row in selected])
        sd = np.array([float(row["sd_changed_area_pct_upper_roi"]) for row in selected])
        label = group.replace("_", " ").title()
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, color=colors(index), label=label)
        ax.fill_between(x, np.maximum(0, y - sd), y + sd, color=colors(index), alpha=.10, linewidth=0)
    ax.axvline(1, color="0.4", linestyle="--", linewidth=1, alpha=.65)
    ax.set(title=args.title, xlabel="Intervention alpha", ylabel="Mean changed area in landmark region (%)")
    ax.set_xticks(sorted({int(row["alpha"]) for row in rows}))
    ax.grid(True, alpha=.22)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
