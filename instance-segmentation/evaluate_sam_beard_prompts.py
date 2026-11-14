#!/usr/bin/env python3
"""Compare two manually prompted SAM proposals on one beard-focused crop.

This is a qualitative prompt probe, not a beard-accuracy evaluation: SAM is
class-agnostic and no reviewed ground-truth beard mask is available yet.
Run with the SAM checkout's virtual environment:
  segment-anything/.venv/bin/python instance-segmentation/evaluate_sam_beard_prompts.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/n/fs/goose")
SAM_ROOT = ROOT / "segment-anything"
sys.path.insert(0, str(SAM_ROOT))
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

DEFAULT_IMAGE = ROOT / (
    "instance-segmentation/results/feature_4228/beard_annotation_pilot/images/"
    "indian_man__seed030__alpha_p7.png"
)
DEFAULT_CHECKPOINT = SAM_ROOT / "checkpoints/sam_vit_b_01ec64.pth"
DEFAULT_OUT = ROOT / "instance-segmentation/results/feature_4228/sam_beard/refined_prompt_probe"


def prompts(width: int, height: int):
    """Two prompt sets: broad lower-face box and tighter chin/beard box."""
    scale = np.array([width, height], dtype=np.float32)
    return {
        "broad_lower_face": {
            "box": np.array([.02, .42, .98, .99], dtype=np.float32) * np.array([width, height, width, height]),
            "points": np.array([
                [.10, .78], [.28, .88], [.50, .92], [.72, .88], [.90, .78],
                [.50, .47], [.06, .39], [.94, .39],
            ], dtype=np.float32) * scale,
            "labels": np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32),
        },
        "tight_beard_box": {
            "box": np.array([.01, .53, .99, .99], dtype=np.float32) * np.array([width, height, width, height]),
            "points": np.array([
                [.10, .78], [.28, .88], [.50, .92], [.72, .88], [.90, .78],
                [.50, .48], [.06, .43], [.94, .43],
            ], dtype=np.float32) * scale,
            "labels": np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32),
        },
    }


def tint_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    red[..., 1] = 55
    out[mask] = .48 * out[mask] + .52 * red[mask]
    return out.astype(np.uint8)


def bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if not len(xs):
        return ("", "", "", "")
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = rgb.shape[:2]
    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    predictor = SamPredictor(model)
    predictor.set_image(rgb)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = [Image.fromarray(rgb)]
    for name, prompt in prompts(width, height).items():
        masks, scores, _ = predictor.predict(
            point_coords=prompt["points"], point_labels=prompt["labels"],
            box=prompt["box"], multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask = masks[best]
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / f"{name}_mask.png")
        Image.fromarray(tint_overlay(rgb, mask)).save(args.out / f"{name}_overlay.png")
        panels.append(Image.fromarray(tint_overlay(rgb, mask)))
        x0, y0, x1, y1 = bbox(mask)
        rows.append({
            "prompt_set": name, "mask_pixels": int(mask.sum()),
            "crop_area_pct": 100 * float(mask.mean()),
            "sam_predicted_iou": float(scores[best]),
            "mask_x0": x0, "mask_y0": y0, "mask_x1": x1, "mask_y1": y1,
        })

    with (args.out / "prompt_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panel_w = max(380, width)
    panel_h = max(340, height + 28)
    montage = Image.new("RGB", (panel_w * len(panels), panel_h), "white")
    draw = ImageDraw.Draw(montage)
    labels = ["input crop", "broad lower-face prompts", "tight beard-box prompts"]
    for i, (panel, label) in enumerate(zip(panels, labels)):
        panel = panel.resize((panel_w, panel_h - 28))
        montage.paste(panel, (i * panel_w, 28))
        draw.text((i * panel_w + 8, 7), label, fill="black")
    montage.save(args.out / "prompt_comparison.png")
    (args.out / "README.md").write_text(
        "# SAM prompt probe on a beard-focused crop\n\n"
        f"Input: `{args.image}`. SAM ViT-B was run with two manually chosen prompt sets. "
        "Red overlays show SAM's top-scoring proposal for each set. This is a qualitative "
        "prompt comparison only: there is no hand-labeled reference mask, so the SAM "
        "predicted IoU is not beard accuracy. Review the overlays before considering "
        "any mask as a beard result.\n\n"
        "- [Prompt comparison](prompt_comparison.png)\n"
        "- [Prompt metrics](prompt_metrics.csv)\n"
    )
    print(f"Saved prompt comparison and metrics to {args.out}")


if __name__ == "__main__":
    main()
