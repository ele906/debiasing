#!/usr/bin/env python3
"""Prepare a small Feature 4228 image sample for manual beard annotation."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/n/fs/goose")
SOURCE = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_4228/images"
LANDMARKER_DIR = ROOT / "face-landmarker"
MODEL = LANDMARKER_DIR / "models/face_landmarker.task"
DEFAULT_OUT = Path("/tmp/goose_beard_annotation_pilot")
GROUPS = (
    "black_man", "black_woman", "east_asian_man", "east_asian_woman",
    "hispanic_man", "hispanic_woman", "indian_man", "indian_woman",
    "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman",
)
DEFAULT_ALPHAS = (-7, 1, 7)


def alpha_code(alpha: int) -> str:
    return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"


def create_landmarker():
    """Load the same local MediaPipe task/model used by face-landmarker reports."""
    sys.path.insert(0, str(LANDMARKER_DIR / ".deps"))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)), num_faces=1
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def lower_face_crop(image: Image.Image, landmarks) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop cheeks/jaw/chin using detected face landmarks; this is not a beard mask."""
    width, height = image.size
    points = np.asarray([[p.x * width, p.y * height] for p in landmarks], dtype=float)
    left, right = points[234], points[454]
    chin, brow = points[152], points[10]
    face_width = max(right[0] - left[0], 1.0)
    face_height = max(chin[1] - brow[1], 1.0)
    cheekline_y = (left[1] + right[1]) / 2

    # Leave the upper face out of the annotation canvas, while retaining the
    # cheeks, mouth, jaw, and chin where visible beard pixels may occur.
    x0 = int(np.floor(left[0] - .10 * face_width))
    x1 = int(np.ceil(right[0] + .10 * face_width))
    y0 = int(np.floor(cheekline_y - .05 * face_height))
    y1 = int(np.ceil(chin[1] + .22 * face_height))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Face Landmarker produced an invalid lower-face crop")
    return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[30])
    parser.add_argument("--alphas", nargs="+", type=int, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not MODEL.is_file():
        raise FileNotFoundError(f"Face Landmarker model is missing: {MODEL}")
    image_dir = args.out / "images"
    context_dir = args.out / "context_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    mp, detector = create_landmarker()
    try:
        for group in args.groups:
            for seed in args.seeds:
                for alpha in args.alphas:
                    source = SOURCE / group / f"seed{seed:03d}_alpha_{alpha_code(alpha)}.png"
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    filename = f"{group}__seed{seed:03d}__alpha_{alpha_code(alpha)}.png"
                    dest = image_dir / filename
                    context_dest = context_dir / filename
                    if (dest.exists() or context_dest.exists()) and not args.overwrite:
                        raise FileExistsError(f"{dest} already exists; use --overwrite to replace it")
                    image = Image.open(source).convert("RGB")
                    result = detector.detect(mp.Image.create_from_file(str(source)))
                    if not result.face_landmarks:
                        raise RuntimeError(f"Face Landmarker found no face in {source}")
                    crop, box = lower_face_crop(image, result.face_landmarks[0])
                    crop.save(dest)
                    # Save a context copy with the exact crop boundary marked.
                    context = image.copy()
                    draw = ImageDraw.Draw(context)
                    draw.rectangle(box, outline=(255, 40, 40), width=max(2, image.width // 300))
                    context.save(context_dest)
                    manifest_rows.append({
                        "image": f"images/{filename}",
                        "context_image": f"context_images/{filename}",
                        "annotation": f"images/{Path(filename).stem}.json",
                        "mask": f"masks/{Path(filename).stem}_beard.png",
                        "crop_x0": box[0], "crop_y0": box[1],
                        "crop_x1": box[2], "crop_y1": box[3],
                        "category": group,
                        "seed": seed,
                        "alpha": alpha,
                        "label_status": "needs_annotation",
                        "source_relative_to_repo": str(source.relative_to(ROOT)),
                    })
    finally:
        detector.close()

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.out / "labels.txt").write_text("beard\n")
    (args.out / "ANNOTATION_GUIDE.md").write_text(
        "# Feature 4228 beard-labeling pilot\n\n"
        f"Prepared {len(manifest_rows)} images ({len(args.groups)} categories, "
        f"{len(args.seeds)} seed(s), alpha values {', '.join(f'{a:+d}' for a in args.alphas)}).\n\n"
        "## Labeling rule\n\n"
        "The annotation images are crops bounded from Face Landmarker cheek/jaw/chin "
        "keypoints. They show the lower face only to make beard labeling easier; the "
        "crop is not a beard segmentation. Use the matching full image in "
        "`context_images/` if you need context. Draw polygons only around visible "
        "beard hair on cheeks, jaw, and chin. Exclude skin, lips, scalp hair, clothing, "
        "and the separate mustache. Do not outline the face or fill the entire crop. "
        "If no beard is visible, save a reviewed annotation with no beard polygon. "
        "Use the same boundary rule at every alpha.\n\n"
        "## LabelMe\n\n"
        "Install LabelMe in the environment where you can use a graphical desktop, "
        "then run from the repository root:\n\n"
        "```bash\n"
        "python -m pip install labelme\n"
        "```\n\n"
        "Open the copied pilot images with:\n\n"
        "```bash\n"
        f"labelme --nodata --labels {args.out / 'labels.txt'} {image_dir}\n"
        "```\n\n"
        "Save polygon shapes with the exact label `beard`; JSON files should be saved "
        "beside their matching images. After labeling, export binary masks with:\n\n"
        "```bash\n"
        f"python instance-segmentation/export_labelme_beard_masks.py --pilot {args.out}\n"
        "```\n\n"
        "These hand-labeled masks will be ground truth for a small transfer-learning "
        "pilot. The failed generic-hair and first SAM masks are not used as labels.\n"
    )
    print(f"Prepared {len(manifest_rows)} images at {args.out}")
    print(f"Images for annotation: {image_dir}")


if __name__ == "__main__":
    main()
