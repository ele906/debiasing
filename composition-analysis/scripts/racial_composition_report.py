"""
Turn the per-iteration composition CSV from racial_composition_analyze.py
into a markdown report summarizing the FINAL-iteration racial composition
per prompt (i.e. the distribution in each prompt's best_image.png-equivalent
generations across seeds), plus a bar chart comparing prompts.

Usage:
    python scripts/racial_composition_report.py \\
        --csv outputs/racial_composition_baseline/composition.csv \\
        --out_md outputs/racial_composition_baseline/report.md \\
        --out_plot outputs/racial_composition_baseline/plots/final_composition.png
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    parser.add_argument("--out_plot", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # final iteration per prompt = max iteration with data for that prompt
    final_rows = []
    for prompt, g in df.groupby("prompt"):
        last_iter = g["iteration"].max()
        final_rows.append(g[g["iteration"] == last_iter])
    final_df = pd.concat(final_rows, ignore_index=True)

    pivot = final_df.pivot_table(
        index="prompt", columns="race", values="pct_of_faces", fill_value=0.0
    )

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("# Racial composition report (baseline, no debias)\n\n")
        f.write(f"Source CSV: `{args.csv}`\n\n")
        f.write(
            "Composition is measured on the final saved optimization "
            "iteration for each (prompt, seed), as classified by FairFace. "
            "Values are the fraction of detected faces in that race "
            "category, averaged across seeds.\n\n"
        )
        f.write("## Final-iteration composition by prompt\n\n")
        f.write(pivot.round(3).to_markdown())
        f.write("\n\n## Overall (averaged across prompts)\n\n")
        f.write(pivot.mean(axis=0).round(3).to_frame("mean_fraction").to_markdown())
        f.write("\n")

    print(f"wrote {args.out_md}")
    print(pivot.round(3))

    if args.out_plot:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(args.out_plot) or ".", exist_ok=True)
        ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 5))
        ax.set_ylabel("Fraction of detected faces")
        ax.set_title("Final-iteration racial composition by prompt (baseline)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150)
        plt.close()
        print(f"wrote {args.out_plot}")


if __name__ == "__main__":
    main()
