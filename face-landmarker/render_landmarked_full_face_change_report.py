#!/usr/bin/env python3
"""Render full-face, landmark-constrained alpha-sweep change reports.

Face Landmarker runs on every generated alpha image.  Its full face-oval mesh
becomes a soft mask; no lower-face-only region is used.  Jet intensity is the
fraction of seeds whose masked pixel changed from its seed baseline, while the
cyan contour marks the 50% face-mask boundary across seeds.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path("/n/fs/goose")
DEFAULT_ROOT = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001"
DEPS = ROOT / "face-landmarker/.deps"
MODEL = ROOT / "face-landmarker/models/face_landmarker.task"
ALPHAS = (-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7)
GROUPS = ("black_man", "black_woman", "east_asian_man", "east_asian_woman", "hispanic_man", "hispanic_woman", "indian_man", "indian_woman", "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman")
# MediaPipe's documented Face Mesh oval, ordered around the complete face.
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


def slug(alpha: int) -> str: return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"
def name(group: str) -> str: return group.replace("_", " ").title()
def load(path: Path) -> np.ndarray: return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
def save(path: Path, value: np.ndarray) -> None: Image.fromarray(np.uint8(np.clip(value, 0, 255))).save(path)


def detector():
    sys.path.append(str(DEPS))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL)), num_faces=1)
    return mp, vision.FaceLandmarker.create_from_options(options)


def face_mask(landmarks, width: int, height: int) -> np.ndarray:
    points = [(landmarks[index].x * width, landmarks[index].y * height) for index in FACE_OVAL]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    return np.asarray(canvas.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0


def grid(path: Path, reference: np.ndarray, maps: dict[int, np.ndarray], faces: dict[int, np.ndarray], cmap: str, title: str, frequency: bool) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    vmax = 1.0 if frequency else max(float(np.percentile(value, 99.5)) for value in maps.values()) or 1.0
    for axis, alpha in zip(axes.flat, ALPHAS):
        value, mask = maps[alpha], faces[alpha]
        axis.imshow(reference.astype(np.uint8))
        axis.imshow(value, cmap=cmap, vmin=0, vmax=vmax, alpha=np.clip(value / vmax * 0.8, 0, 0.8))
        if float(mask.max()) >= .5: axis.contour(mask, levels=[.5], colors="#00eaff", linewidths=1.2)
        axis.set_title(f"{alpha:+d}", fontsize=10); axis.axis("off")
    axes.flat[len(ALPHAS)].axis("off")
    figure.suptitle(title, y=.99); figure.tight_layout(rect=(0, 0, 1, .965)); figure.savefig(path, dpi=150); plt.close(figure)


def group_report(run: Path, group: str, out: Path, mp, landmarker, threshold: float, feature: str) -> list[dict[str, object]]:
    paths = sorted((run / "images" / group / "baseline").glob("seed*.png"))
    seeds = [int(re.search(r"seed(\d+)", path.name).group(1)) for path in paths]
    baselines = [load(path) for path in paths]
    reference = np.mean(baselines, axis=0)
    target = out / group; target.mkdir(parents=True); save(target / "average_reference.png", reference)
    frequencies, attenuations, face_maps, rows = {}, {}, {}, []
    for alpha in ALPHAS:
        masks, changes, blurs, changed_area, mae_values = [], [], [], [], []
        for seed, baseline in zip(seeds, baselines):
            altered_path = run / "images" / group / f"seed{seed:03d}_alpha_{slug(alpha)}.png"
            altered = load(altered_path); result = landmarker.detect(mp.Image.create_from_file(str(altered_path)))
            if not result.face_landmarks: continue
            mask = face_mask(result.face_landmarks[0], altered.shape[1], altered.shape[0])
            mae = np.abs(altered - baseline).mean(axis=2)
            change = (mae >= threshold).astype(np.float32) * mask
            masks.append(mask); changes.append(change)
            blurs.append(np.asarray(Image.fromarray(np.uint8(np.clip(mae, 0, 255))).filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) * mask)
            area = float(mask.sum()); changed_area.append(float(change.sum() / area * 100)); mae_values.append(float((mae * mask).sum() / area))
        if not masks: raise RuntimeError(f"No faces detected for {group} alpha {alpha:+d}")
        frequencies[alpha], attenuations[alpha], face_maps[alpha] = np.mean(changes, axis=0), np.mean(blurs, axis=0), np.mean(masks, axis=0)
        rows.append({"group": group, "alpha": alpha, "n_landmarks": len(masks), "threshold_rgb_mae": threshold, "mean_changed_area_pct_full_face": float(np.mean(changed_area)), "sd_changed_area_pct_full_face": float(np.std(changed_area, ddof=1)), "mean_rgb_mae_full_face": float(np.mean(mae_values))})
    grid(target / "alpha_full_face_change_frequency_jet.png", reference, frequencies, face_maps, "jet", f"{feature} · {name(group)} · full-face landmarked change frequency", True)
    grid(target / "alpha_blurred_mean_pixel_change.png", reference, attenuations, face_maps, "magma", f"{feature} · {name(group)} · full-face blurred mean pixel change", False)
    return rows


def write_readme(out: Path, rows: list[dict[str, object]], feature: str, threshold: float) -> None:
    indexed = {group: {int(row["alpha"]): row for row in rows if row["group"] == group} for group in GROUPS}
    lines = [f"# {feature} — landmark-constrained full-face change report", "", "MediaPipe Face Landmarker runs on every generated alpha image. Its complete face-oval mesh forms a soft full-face mask; no lower-face-only region is used. The cyan line in each grid is the 50% mask-frequency contour.", "", f"Jet intensity is the fraction of seeds whose pixel changes by RGB MAE ≥ **{threshold:.1f}** from its seed baseline, within that alpha image's full-face mask. This establishes spatially repeated visual change, not a semantic label such as forehead covering.", "", "## Mean changed full-face area by alpha", "", "| Category | " + " | ".join(f"{alpha:+d}" for alpha in ALPHAS) + " |", "|---|" + "|".join("---:" for _ in ALPHAS) + "|"]
    for group in GROUPS:
        lines.append("| " + name(group) + " | " + " | ".join(f"{indexed[group][alpha]['mean_changed_area_pct_full_face']:.1f}% ± {indexed[group][alpha]['sd_changed_area_pct_full_face']:.1f}" for alpha in ALPHAS) + " |")
    lines += ["", "- `full_face_change_metrics.csv` — tidy data.", "- `<category>/alpha_full_face_change_frequency_jet.png` — full-face change-frequency maps.", "- `<category>/alpha_blurred_mean_pixel_change.png` — masked attenuation maps."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("feature", type=int); parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT); parser.add_argument("--out", type=Path); parser.add_argument("--change-threshold", type=float, default=12.0); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); run = args.run_root / f"feature_{args.feature}"; out = args.out or run / "full_face_landmarker_change_report"
    if out.exists():
        if not args.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True); mp, landmarker = detector(); rows = []
    try:
        for group in GROUPS:
            print(f"Rendering {group}", flush=True); rows.extend(group_report(run, group, out, mp, landmarker, args.change_threshold, f"Feature {args.feature}"))
    finally: landmarker.close()
    with (out / "full_face_change_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_readme(out, rows, f"Feature {args.feature}", args.change_threshold); print(f"Wrote {out}", flush=True)


if __name__ == "__main__": main()
