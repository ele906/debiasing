#!/usr/bin/env python3
"""Render unconstrained, full-image alpha-sweep change reports.

This deliberately does not impose a lower-face or other hand-drawn region.
The jet map is the fraction of seeds whose pixel changes from its baseline by
at least the RGB-MAE threshold; the companion map is blurred mean RGB change.
Neither map assigns a semantic label such as beard or head covering.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


ROOT = Path("/n/fs/goose")
DEFAULT_ROOT = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001"
ALPHAS = (-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7)
GROUPS = (
    "black_man", "black_woman", "east_asian_man", "east_asian_woman",
    "hispanic_man", "hispanic_woman", "indian_man", "indian_woman",
    "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman",
)


def slug(alpha: int) -> str:
    return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"


def label(group: str) -> str:
    return group.replace("_", " ").title()


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def save(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(value, 0, 255))).save(path)


def blurred(value: np.ndarray, radius: float = 6.0) -> np.ndarray:
    return np.asarray(Image.fromarray(np.uint8(np.clip(value, 0, 255))).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)


def render_grid(path: Path, reference: np.ndarray, maps: dict[int, np.ndarray], cmap: str, title: str, is_frequency: bool) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    vmax = 1.0 if is_frequency else max(float(np.percentile(item, 99.5)) for item in maps.values()) or 1.0
    for axis, alpha in zip(axes.flat, ALPHAS):
        value = maps[alpha]
        axis.imshow(reference.astype(np.uint8))
        axis.imshow(value, cmap=cmap, vmin=0, vmax=vmax, alpha=np.clip(value / vmax * 0.80, 0, 0.80))
        axis.set_title(f"{alpha:+d}", fontsize=10)
        axis.axis("off")
    axes.flat[len(ALPHAS)].axis("off")
    figure.suptitle(title, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_group(run: Path, group: str, out: Path, threshold: float, feature: str) -> list[dict[str, object]]:
    base_paths = sorted((run / "images" / group / "baseline").glob("seed*.png"))
    seeds = [int(re.search(r"seed(\d+)", path.name).group(1)) for path in base_paths]
    baselines = [image(path) for path in base_paths]
    reference = np.mean(baselines, axis=0)
    group_out = out / group
    group_out.mkdir(parents=True)
    save(group_out / "average_reference.png", reference)
    frequency: dict[int, np.ndarray] = {}
    attenuation: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for alpha in ALPHAS:
        changed, softened, areas, maes = [], [], [], []
        for seed, baseline in zip(seeds, baselines):
            altered = image(run / "images" / group / f"seed{seed:03d}_alpha_{slug(alpha)}.png")
            mae = np.abs(altered - baseline).mean(axis=2)
            mask = (mae >= threshold).astype(np.float32)
            changed.append(mask)
            softened.append(blurred(mae))
            areas.append(float(mask.mean() * 100.0))
            maes.append(float(mae.mean()))
        frequency[alpha] = np.mean(changed, axis=0)
        attenuation[alpha] = np.mean(softened, axis=0)
        rows.append({
            "group": group, "alpha": alpha, "n_seeds": len(seeds),
            "change_threshold_rgb_mae": threshold,
            "mean_changed_area_pct_full_image": float(np.mean(areas)),
            "sd_changed_area_pct_full_image": float(np.std(areas, ddof=1)),
            "mean_rgb_mae_full_image": float(np.mean(maes)),
        })
    render_grid(group_out / "alpha_full_image_change_frequency_jet.png", reference, frequency, "jet",
                f"{feature} · {label(group)} · full-image change frequency across {len(seeds)} seeds", True)
    render_grid(group_out / "alpha_blurred_mean_pixel_change.png", reference, attenuation, "magma",
                f"{feature} · {label(group)} · full-image blurred mean pixel change", False)
    return rows


def write_report(out: Path, rows: list[dict[str, object]], feature: str, threshold: float) -> None:
    by_group = {group: {int(row["alpha"]): row for row in rows if row["group"] == group} for group in GROUPS}
    lines = [
        f"# {feature} — unconstrained full-image change report", "",
        "No lower-face or hand-drawn facial subregion is used. For each seed and alpha, a pixel is counted when its RGB MAE from that seed's baseline is at least "
        f"**{threshold:.1f}** (0–255 scale). The jet maps show the fraction of seeds in which each full-image pixel crossed that threshold.", "",
        "This can establish spatially consistent visual change, including changes outside the face. It does **not** identify beard, hair, hat, hijab, or any other semantic object; those claims require a semantic detector.", "",
        "## Mean changed full-image area by alpha", "",
        "Each cell is the mean changed full-image area followed by across-seed SD. `+1` is the reference reconstruction.", "",
        "| Category | " + " | ".join(f"{alpha:+d}" for alpha in ALPHAS) + " |",
        "|---|" + "|".join("---:" for _ in ALPHAS) + "|",
    ]
    for group in GROUPS:
        values = [f"{by_group[group][alpha]['mean_changed_area_pct_full_image']:.1f}% ± {by_group[group][alpha]['sd_changed_area_pct_full_image']:.1f}" for alpha in ALPHAS]
        lines.append("| " + label(group) + " | " + " | ".join(values) + " |")
    lines += ["", "## Files", "", "- `full_image_change_metrics.csv` — tidy measurements.", "- `<category>/alpha_full_image_change_frequency_jet.png` — seed-consistency overlay.", "- `<category>/alpha_blurred_mean_pixel_change.png` — attenuation/change overlay."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature", type=int)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--change-threshold", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run_root / f"feature_{args.feature}"
    out = args.out or run / "full_image_change_report"
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    feature = f"Feature {args.feature}"
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        print(f"Rendering {group}", flush=True)
        rows.extend(run_group(run, group, out, args.change_threshold, feature))
    with (out / "full_image_change_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_report(out, rows, feature, args.change_threshold)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
