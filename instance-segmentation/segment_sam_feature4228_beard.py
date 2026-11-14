#!/usr/bin/env python3
"""Create a prompt-guided SAM mask for one Feature 4228 beard image.

SAM is class-agnostic: the prompt points below identify beard/moustache pixels
as foreground and adjacent skin as background. Review the overlay before using
the result as a beard-area measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / (
    "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/"
    "feature_4228/images/indian_man/seed053_alpha_p7.png"
)
DEFAULT_CHECKPOINT = ROOT / "instance-segmentation/.cache/sam_vit_b_01ec64.pth"
DEFAULT_OUT = ROOT / "instance-segmentation/results/feature_4228/sam_beard"

# Tuned by visually inspecting DEFAULT_IMAGE (512 x 512). The box excludes
# forehead/turban; positive points cover beard hairs; negative points mark
# skin and mouth that must remain outside the mask.
BOX = np.array([145, 305, 370, 490], dtype=np.float32)
FOREGROUND = np.array(
    [[210, 350], [302, 350], [220, 400], [256, 415], [292, 400]], dtype=np.float32
)
BACKGROUND = np.array(
    [[256, 278], [186, 315], [326, 315], [256, 370], [166, 380], [348, 380]],
    dtype=np.float32,
)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    tint = np.zeros_like(result)
    tint[..., 0] = 255
    tint[..., 1] = 70
    result[mask] = 0.5 * result[mask] + 0.5 * tint[mask]
    return result.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {args.checkpoint}. Download ViT-B from "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)))
    predictor.set_image(rgb)
    points = np.concatenate((FOREGROUND, BACKGROUND))
    labels = np.concatenate((np.ones(len(FOREGROUND)), np.zeros(len(BACKGROUND))))
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, box=BOX, multimask_output=True
    )
    best = int(np.argmax(scores))
    mask = masks[best]

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray(np.uint8(mask) * 255).save(args.out / f"{stem}_beard_mask.png")
    Image.fromarray(overlay(rgb, mask)).save(args.out / f"{stem}_beard_overlay.png")
    np.save(args.out / f"{stem}_beard_mask.npy", mask)
    (args.out / f"{stem}_metrics.txt").write_text(
        f"image={args.image}\nmask_pixels={int(mask.sum())}\npredicted_iou={float(scores[best]):.4f}\n"
    )
    print(f"Saved {stem}: {int(mask.sum())} mask pixels; predicted IoU {scores[best]:.4f}")


if __name__ == "__main__":
    main()
