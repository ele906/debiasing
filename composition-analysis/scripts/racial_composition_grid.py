"""
Visual sanity-check grid: for each prompt, show N sample seeds' images at a
few checkpoints across the 50 optimization iterations (default: 1st, 20th,
40th, 50th), so you can eyeball how each run's subject drifts from init to
final -- before/alongside the FairFace race-distribution analysis in
racial_composition_analyze.py.

Usage:
    python scripts/racial_composition_grid.py \\
        --data_dir outputs/racial_composition \\
        --n_seeds_to_show 5 \\
        --iterations 1 20 40 50 \\
        --out_dir outputs/racial_composition/plots
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_seeds_to_show", type=int, default=5)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[1, 20, 40, 50],
        help="1-indexed iteration numbers to show (1 = first saved frame, "
        "50 = last, assuming n_iters=50).",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/racial_composition/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # saved files are 0-indexed (0.png .. n_iters-1.png)
    iter_indices = [i - 1 for i in args.iterations]

    prompt_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(d)
    )
    for prompt_dir in prompt_dirs:
        prompt_slug = os.path.basename(prompt_dir)
        seed_dirs = sorted(
            glob.glob(os.path.join(prompt_dir, "seed*")),
            key=lambda p: int(os.path.basename(p).replace("seed", "")),
        )[: args.n_seeds_to_show]
        if not seed_dirs:
            continue

        n_rows = len(seed_dirs)
        n_cols = len(iter_indices)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
        )
        for row, seed_dir in enumerate(seed_dirs):
            seed_label = os.path.basename(seed_dir)
            for col, (it_num, it_idx) in enumerate(zip(args.iterations, iter_indices)):
                ax = axes[row][col]
                img_path = os.path.join(seed_dir, f"{it_idx}.png")
                if os.path.exists(img_path):
                    ax.imshow(Image.open(img_path))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"iter {it_num}")
                if col == 0:
                    ax.set_ylabel(seed_label, rotation=0, ha="right", va="center")

        fig.suptitle(prompt_slug.replace("_", " "))
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{prompt_slug}_grid.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
