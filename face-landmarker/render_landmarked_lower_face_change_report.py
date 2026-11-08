#!/usr/bin/env python3
"""Render landmark-derived beard-region and lower-face change reports.

For every generated alpha image, MediaPipe Face Landmarker supplies a
lower-face / extended-chin polygon. Averaging those per-seed polygons produces
the jet map: a pixel's intensity is the fraction of seeds for which the
landmarker selected it as part of that beard-region geometry. This is not a
semantic beard-segmentation claim.

The script is self-contained apart from MediaPipe and its Face Landmarker task
file.  The workspace-local dependency/model locations are the defaults used by
this experiment, but both can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path("/n/fs/goose")
DEFAULT_RUN = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_4228"
DEFAULT_DEPS = ROOT / "face-landmarker/.deps"
DEFAULT_MODEL = ROOT / "face-landmarker/models/face_landmarker.task"
ALPHAS = (-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7)
GROUP_ORDER = (
    "black_man", "black_woman", "east_asian_man", "east_asian_woman",
    "hispanic_man", "hispanic_woman", "indian_man", "indian_woman",
    "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman",
)
BASELINE_RE = re.compile(r"seed(\d+)\.png$")


def alpha_slug(alpha: int) -> str:
    return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"


def display_group(group: str) -> str:
    return group.replace("_", " ").title()


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(array, 0, 255))).save(path)


def soft_polygon_mask(size: tuple[int, int], points: list[tuple[float, float]], blur: float = 2.0) -> np.ndarray:
    """Rasterize a polygon as a soft, anti-aliased-looking ROI mask."""
    width, height = size
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    if blur:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def beard_roi_from_landmarks(landmarks, width: int, height: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build an explainable lower-face loop, extended past the chin for beards.

    The top is anchored at the left/right face contour and the loop's bottom is
    derived from the chin-to-forehead face height.  It deliberately includes
    potential long-beard space below the Face Mesh oval.
    """
    def point(index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array((landmark.x * width, landmark.y * height), dtype=np.float32)

    left, right, chin, forehead = point(234), point(454), point(152), point(10)
    face_height = max(float(chin[1] - forehead[1]), 1.0)
    center_x = float((left[0] + right[0]) / 2.0)
    top_y = float(min(max((left[1] + right[1]) / 2.0 - 0.03 * face_height, 0), height - 1))
    lower_y = min(float(chin[1] + 0.42 * face_height), height - 1)
    # A smooth, landmark-derived trapezoid is more inclusive than the Face Mesh
    # oval, which otherwise clips long beards at landmark 152 (the chin).
    points = [
        (float(left[0]), top_y),
        (float(right[0]), top_y),
        (float(center_x + 0.70 * (right[0] - center_x)), lower_y),
        (center_x, lower_y),
        (float(center_x - 0.70 * (center_x - left[0])), lower_y),
    ]
    return soft_polygon_mask((width, height), points), points


def draw_roi_outline(image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    output = Image.fromarray(np.uint8(np.clip(image, 0, 255))).convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.line(points + [points[0]], fill=(0, 235, 255), width=3, joint="curve")
    return np.asarray(output, dtype=np.float32)


@dataclass
class SeedROI:
    seed: int
    roi: np.ndarray
    outline: list[tuple[float, float]]
    baseline: np.ndarray


def build_detector(model_path: Path, deps_dir: Path):
    # Appending preserves the workspace's NumPy/Matplotlib ABI while locating
    # the isolated MediaPipe package.
    if str(deps_dir) not in sys.path:
        sys.path.append(str(deps_dir))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)), num_faces=1
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def find_seed_rois(group_dir: Path, detector, mp) -> tuple[list[SeedROI], list[int]]:
    rois: list[SeedROI] = []
    failed: list[int] = []
    for path in sorted((group_dir / "baseline").glob("seed*.png")):
        match = BASELINE_RE.search(path.name)
        if not match:
            continue
        seed = int(match.group(1))
        baseline = read_rgb(path)
        result = detector.detect(mp.Image.create_from_file(str(path)))
        if not result.face_landmarks:
            failed.append(seed)
            continue
        height, width = baseline.shape[:2]
        roi, outline = beard_roi_from_landmarks(result.face_landmarks[0], width, height)
        rois.append(SeedROI(seed, roi, outline, baseline))
    return rois, failed


def render_landmarker_frequency_grid(
    path: Path, reference: np.ndarray, maps: dict[int, np.ndarray], title: str
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    panels = axes.flat[:len(ALPHAS)]
    for axis, alpha in zip(panels, ALPHAS):
        axis.imshow(reference.astype(np.uint8))
        overlay = maps[alpha]
        axis.imshow(overlay, cmap="jet", vmin=0, vmax=1, alpha=np.clip(overlay * 0.80, 0, 0.80))
        if float(overlay.max()) >= 0.5:
            axis.contour(overlay, levels=[0.5], colors="#00eaff", linewidths=1.2)
        axis.set_title(f"{alpha:+d}", fontsize=10)
        axis.axis("off")
    axes.flat[len(ALPHAS)].axis("off")
    figure.suptitle(title, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_difference_grid(
    path: Path, reference: np.ndarray, outline: list[tuple[float, float]], maps: dict[int, np.ndarray], title: str
) -> None:
    vmax = max(float(np.percentile(value, 99.5)) for value in maps.values()) or 1.0
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    panels = axes.flat[:len(ALPHAS)]
    for axis, alpha in zip(panels, ALPHAS):
        axis.imshow(draw_roi_outline(reference, outline).astype(np.uint8))
        value = maps[alpha]
        axis.imshow(value, cmap="magma", vmin=0, vmax=vmax, alpha=np.clip(value / vmax * 0.86, 0, 0.86))
        axis.set_title(f"{alpha:+d}", fontsize=10)
        axis.axis("off")
    legend_axis = axes.flat[len(ALPHAS)]
    scalar = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    figure.colorbar(scalar, cax=legend_axis, label="mean RGB MAE (0–255)")
    figure.suptitle(title, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report_group(group: str, group_dir: Path, out_dir: Path, detector, mp, threshold: float) -> list[dict[str, object]]:
    rois, failed = find_seed_rois(group_dir, detector, mp)
    if not rois:
        raise RuntimeError(f"No faces found in {group}")
    reference = np.mean([entry.baseline for entry in rois], axis=0)
    mean_roi = np.mean([entry.roi for entry in rois], axis=0)
    mean_outline = rois[0].outline  # shown on an average image as a representative seed geometry.
    group_out = out_dir / group
    group_out.mkdir(parents=True, exist_ok=True)
    write_image(group_out / "average_reference.png", draw_roi_outline(reference, mean_outline))
    write_image(group_out / "average_landmark_roi.png", np.dstack([mean_roi * 0, mean_roi * 175, mean_roi * 255]))

    landmark_frequency: dict[int, np.ndarray] = {}
    blur_difference: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for alpha in ALPHAS:
        per_seed_area: list[float] = []
        per_seed_fraction: list[float] = []
        per_seed_mae: list[float] = []
        changed_maps: list[np.ndarray] = []
        blurred_maps: list[np.ndarray] = []
        landmark_masks: list[np.ndarray] = []
        landmark_areas: list[float] = []
        failed_alpha_landmarks: list[int] = []
        for entry in rois:
            altered_path = group_dir / f"seed{entry.seed:03d}_alpha_{alpha_slug(alpha)}.png"
            if not altered_path.exists():
                raise FileNotFoundError(altered_path)
            altered = read_rgb(altered_path)
            alpha_landmarks = detector.detect(mp.Image.create_from_file(str(altered_path))).face_landmarks
            if alpha_landmarks:
                height, width = altered.shape[:2]
                alpha_roi, _ = beard_roi_from_landmarks(alpha_landmarks[0], width, height)
                landmark_masks.append(alpha_roi)
                landmark_areas.append(float(alpha_roi.sum()))
            else:
                failed_alpha_landmarks.append(entry.seed)
            mae = np.abs(altered - entry.baseline).mean(axis=2)
            changed = (mae >= threshold).astype(np.float32) * entry.roi
            blurred = np.asarray(
                Image.fromarray(np.uint8(np.clip(mae, 0, 255))).filter(ImageFilter.GaussianBlur(6)), dtype=np.float32
            ) * entry.roi
            roi_area = float(entry.roi.sum())
            per_seed_area.append(float(changed.sum()))
            per_seed_fraction.append(float(changed.sum() / roi_area * 100.0))
            per_seed_mae.append(float((mae * entry.roi).sum() / roi_area))
            changed_maps.append(changed)
            blurred_maps.append(blurred)
        if not landmark_masks:
            raise RuntimeError(f"No Face Landmarker detections for {group} alpha {alpha:+d}")
        landmark_frequency[alpha] = np.mean(landmark_masks, axis=0)
        blur_difference[alpha] = np.mean(blurred_maps, axis=0)
        rows.append({
            "group": group,
            "alpha": alpha,
            "n_seeds": len(rois),
            "failed_landmarks": ";".join(map(str, failed)),
            "n_alpha_landmarks": len(landmark_masks),
            "failed_alpha_landmarks": ";".join(map(str, failed_alpha_landmarks)),
            "change_threshold_rgb_mae": threshold,
            "mean_landmark_roi_area_px": float(mean_roi.sum()),
            "mean_alpha_landmark_beard_region_area_px": float(np.mean(landmark_areas)),
            "sd_alpha_landmark_beard_region_area_px": float(np.std(landmark_areas, ddof=1)) if len(landmark_areas) > 1 else 0.0,
            "mean_changed_area_px": float(np.mean(per_seed_area)),
            "sd_changed_area_px": float(np.std(per_seed_area, ddof=1)) if len(per_seed_area) > 1 else 0.0,
            "mean_changed_area_pct_roi": float(np.mean(per_seed_fraction)),
            "sd_changed_area_pct_roi": float(np.std(per_seed_fraction, ddof=1)) if len(per_seed_fraction) > 1 else 0.0,
            "mean_rgb_mae_within_roi": float(np.mean(per_seed_mae)),
        })
    render_landmarker_frequency_grid(
        group_out / "alpha_landmark_beard_region_frequency_jet.png", reference, landmark_frequency,
        f"Feature 4228 · {display_group(group)} · landmark beard-region selection frequency across {len(rois)} seeds",
    )
    render_difference_grid(
        group_out / "alpha_blurred_mean_pixel_change.png", reference, mean_outline, blur_difference,
        f"Feature 4228 · {display_group(group)} · blurred mean pixel change in landmark ROI",
    )
    return rows


def write_markdown(out_dir: Path, rows: list[dict[str, object]], threshold: float) -> None:
    by_group = {group: {int(row["alpha"]): row for row in rows if row["group"] == group} for group in GROUP_ORDER}
    lines = [
        "# Feature 4228 — landmark-constrained lower-face change report",
        "",
        "For each alpha image and seed, MediaPipe Face Landmarker constructs a lower-face/extended-chin polygon. The jet overlay is the pixelwise mean of those polygons: its intensity is the fraction of seeds for which the landmarker selected that pixel as part of the beard-region geometry. This is not a semantic beard segmenter.",
        "",
        "The cyan contour in each jet panel marks 50% selection frequency. The geometry extends below landmark 152 (chin) so long beards are not clipped. The `alpha_blurred_mean_pixel_change.png` panel is separate attenuation evidence: seed-level RGB-MAE maps are blurred before averaging. Its pixel-change threshold is "
        f"**{threshold:.1f}** (0–255 scale).",
        "",
        "## Mean landmark beard-region area by alpha",
        "",
        "Each cell is mean landmark-selected beard-region area, normalized to the `+1` reference's mean area, followed by its across-seed standard deviation. Unlike the old change map, `+1` is expected to show a nonzero region here.",
        "",
        "| Category | " + " | ".join(f"{alpha:+d}" for alpha in ALPHAS) + " |",
        "|---|" + "|".join("---:" for _ in ALPHAS) + "|",
    ]
    for group in GROUP_ORDER:
        values = []
        for alpha in ALPHAS:
            row = by_group[group][alpha]
            reference_area = by_group[group][1]["mean_alpha_landmark_beard_region_area_px"]
            ratio = float(row["mean_alpha_landmark_beard_region_area_px"]) / float(reference_area) * 100.0
            sd_ratio = float(row["sd_alpha_landmark_beard_region_area_px"]) / float(reference_area) * 100.0
            values.append(f"{ratio:.1f}% ± {sd_ratio:.1f}")
        lines.append("| " + display_group(group) + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Files",
        "",
        "- `beard_region_change_metrics.csv` — tidy raw metrics for plotting or statistical follow-up.",
        "- `<category>/average_reference.png` — category-average baseline with representative ROI loop.",
        "- `<category>/alpha_landmark_beard_region_frequency_jet.png` — blue→green→yellow→red landmarker-selection-frequency overlays; no in-grid legend cell.",
        "- `<category>/alpha_blurred_mean_pixel_change.png` — blurred mean pixel-change/attenuation overlays.",
        "",
        "The jet score is facial geometry selected by landmarks, not a statement that every selected pixel is facial hair. The separate attenuation panel is evidence of intervention-localized RGB change. A later semantic beard model can replace the geometric region while retaining the same report layout.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--deps-dir", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--change-threshold", type=float, default=12.0, help="RGB MAE threshold on the 0–255 scale")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    out_dir = args.out or args.run / "beard_landmarker_report"
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {out_dir}")
        shutil.rmtree(out_dir)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    out_dir.mkdir(parents=True)
    mp, detector = build_detector(args.model, args.deps_dir)
    try:
        rows: list[dict[str, object]] = []
        groups = [group for group in GROUP_ORDER if (args.run / "images" / group).is_dir()]
        for group in groups:
            print(f"Rendering {group}", flush=True)
            rows.extend(report_group(group, args.run / "images" / group, out_dir, detector, mp, args.change_threshold))
    finally:
        detector.close()
    fieldnames = list(rows[0])
    with (out_dir / "beard_region_change_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(out_dir, rows, args.change_threshold)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
