#!/usr/bin/env python3
"""Measure feature changes in a landmark-anchored head-covering region.

The soft quadrilateral extends above the forehead and beyond the temples, then
down to the eyebrow line. It captures the forehead and surrounding upper-head
context where a covering may sit. It is a spatial ROI, not hijab segmentation.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path("/n/fs/goose")
RUN = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_2023"
DEPS = ROOT / "face-landmarker/.deps"
MODEL = ROOT / "face-landmarker/models/face_landmarker.task"
DEFAULT_GROUPS = ("middle_eastern_man", "middle_eastern_woman", "indian_man")
ALL_GROUPS = ("black_man", "black_woman", "east_asian_man", "east_asian_woman", "hispanic_man", "hispanic_woman", "indian_man", "indian_woman", "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman")
ALPHAS = (-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7)


def slug(a: int) -> str: return f"m{abs(a)}" if a < 0 else f"p{a}"
def load(p: Path) -> np.ndarray: return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def setup_detector():
    sys.path.append(str(DEPS))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL)), num_faces=1)
    return mp, vision.FaceLandmarker.create_from_options(options)


def upper_face_mask(landmarks, width: int, height: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    def pt(i):
        p = landmarks[i]
        return (float(p.x * width), float(p.y * height))
    # Use face height to expand above landmark 10 and outside both temples.
    # 105/334 approximate brow height; the lower edge still includes upper face.
    left_temple, right_temple = pt(234), pt(454)
    forehead, brow_left, brow_right = pt(10), pt(105), pt(334)
    brow_y = (brow_left[1] + brow_right[1]) / 2
    face_height = max(pt(152)[1] - forehead[1], 1.0)
    side_expand = (right_temple[0] - left_temple[0]) * 0.22
    top_expand = face_height * 0.42
    polygon = [
        (max(0, left_temple[0] - side_expand), max(0, forehead[1] - top_expand)),
        (min(width - 1, right_temple[0] + side_expand), max(0, forehead[1] - top_expand)),
        (min(width - 1, right_temple[0] + side_expand), min(height - 1, brow_y)),
        (max(0, left_temple[0] - side_expand), min(height - 1, brow_y)),
    ]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    mask = np.asarray(canvas.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255
    return mask, polygon


def render_grid(path: Path, reference: np.ndarray, maps: dict[int, np.ndarray], masks: dict[int, np.ndarray], title: str, cmap: str, frequency: bool):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    vmax = 1.0 if frequency else max(float(np.percentile(v, 99.5)) for v in maps.values()) or 1.0
    for ax, alpha in zip(axes.flat, ALPHAS):
        ax.imshow(reference.astype(np.uint8))
        value = maps[alpha]
        ax.imshow(value, cmap=cmap, vmin=0, vmax=vmax, alpha=np.clip(value / vmax * .8, 0, .8))
        if frequency and masks[alpha].max() >= .5:
            ax.contour(masks[alpha], levels=[.5], colors=["#00eaff"], linewidths=1.2)
        ax.set_title(f"{alpha:+d}"); ax.axis("off")
    axes.flat[len(ALPHAS)].axis("off")
    fig.suptitle(title, y=.99); fig.tight_layout(rect=(0, 0, 1, .965)); fig.savefig(path, dpi=150); plt.close(fig)


def process(group: str, out: Path, mp, detector, threshold: float):
    image_root = RUN / "images" / group
    originals = sorted((image_root / "baseline").glob("seed*.png"))
    seeds = [int(re.search(r"seed(\d+)", p.name).group(1)) for p in originals]
    baselines = [load(p) for p in originals]
    reference = np.mean(baselines, axis=0)
    dest = out / group; dest.mkdir(parents=True); Image.fromarray(np.uint8(reference)).save(dest / "average_reference.png")
    rows, frequencies, changes, mask_means = [], {}, {}, {}
    for alpha in ALPHAS:
        masks, hitmaps, diffs, area_pct, maes = [], [], [], [], []
        for seed, baseline in zip(seeds, baselines):
            p = image_root / f"seed{seed:03d}_alpha_{slug(alpha)}.png"
            edited = load(p)
            result = detector.detect(mp.Image.create_from_file(str(p)))
            if not result.face_landmarks: continue
            mask, _ = upper_face_mask(result.face_landmarks[0], edited.shape[1], edited.shape[0])
            mae = np.abs(edited - baseline).mean(axis=2)
            hit = (mae >= threshold).astype(np.float32) * mask
            masks.append(mask); hitmaps.append(hit)
            diffs.append(np.asarray(Image.fromarray(np.uint8(np.clip(mae, 0, 255))).filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) * mask)
            area = float(mask.sum()); area_pct.append(float(hit.sum() / area * 100)); maes.append(float((mae * mask).sum() / area))
        if not masks: raise RuntimeError(f"No landmarks for {group} alpha {alpha:+d}")
        frequencies[alpha], changes[alpha], mask_means[alpha] = np.mean(hitmaps, axis=0), np.mean(diffs, axis=0), np.mean(masks, axis=0)
        rows.append({"group": group, "alpha": alpha, "n_seeds": len(masks), "change_threshold_rgb_mae": threshold, "mean_upper_roi_area_px": float(mask_means[alpha].sum()), "mean_changed_area_pct_upper_roi": float(np.mean(area_pct)), "sd_changed_area_pct_upper_roi": float(np.std(area_pct, ddof=1)), "mean_rgb_mae_upper_roi": float(np.mean(maes))})
    title = group.replace("_", " ").title()
    render_grid(dest / "alpha_head_covering_region_change_frequency_jet.png", reference, frequencies, mask_means, f"Feature 2023 · {title} · head-covering-region change frequency", "jet", True)
    render_grid(dest / "alpha_head_covering_region_blurred_pixel_change.png", reference, changes, mask_means, f"Feature 2023 · {title} · blurred mean change in head-covering region", "magma", False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--change-threshold", type=float, default=12.0)
    parser.add_argument("--groups", nargs="+", choices=ALL_GROUPS, default=DEFAULT_GROUPS)
    args = parser.parse_args()
    run = args.run
    feature_match = re.search(r"feature_(\d+)$", str(run))
    feature_label = f"Feature {feature_match.group(1)}" if feature_match else "Feature"
    out = args.out or run / "head_covering_region_landmark_report"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    mp, detector = setup_detector(); rows = []
    try:
        for group in args.groups:
            print(f"Rendering {group}", flush=True)
            rows.extend(process(group, out, mp, detector, args.change_threshold))
    finally: detector.close()
    with (out / "upper_face_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = [f"# {feature_label} landmark-anchored head-covering-region analysis", "", "Face Landmarker anchors a soft region that extends above the forehead and beyond the temples, down to the eyebrow line. It includes surrounding head context where a covering may sit. This is a geometric ROI, not hijab or semantic covering detection.", "", f"Jet maps show the fraction of seeds whose pixel changes from baseline by RGB MAE ≥ {args.change_threshold:.1f}, within each image's landmark-anchored region. The cyan contour marks the mean region boundary. Blurred maps show mean pixel change inside the same region.", "", "## Changed area by alpha", "", "| Group | Alpha | Mean area % ± SD | Mean RGB MAE |", "|---|---:|---:|---:|"]
    for row in rows: lines.append(f"| {row['group']} | {row['alpha']:+d} | {row['mean_changed_area_pct_upper_roi']:.1f}% ± {row['sd_changed_area_pct_upper_roi']:.1f} | {row['mean_rgb_mae_upper_roi']:.2f} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__": main()
