#!/usr/bin/env python3
"""Probe whether SCHP human-parsing hair masks include beard pixels.

The ATR model has a generic ``hair`` class, not a beard class. This script
exports that class and its intersection with a Face-Landmarker jaw/chin ROI so
the result can be judged visually before any across-alpha claim is made.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/n/fs/goose")
DEFAULT_IMAGES = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_4228/images"
DEFAULT_OUT = ROOT / "instance-segmentation/results/feature_4228/beard_probe"
DEFAULT_MODEL_CACHE = ROOT / "instance-segmentation/.cache"
DEFAULT_MP_DEPS = ROOT / "face-landmarker/.deps"
DEFAULT_MP_MODEL = ROOT / "face-landmarker/models/face_landmarker.task"
MODEL_URL = "https://huggingface.co/pirocheto/schp-atr-18/resolve/main/onnx/schp-atr-18-int8-static.onnx"
HAIR_CLASS_ID = 2  # SCHP ATR label list: 2 = Hair
ALPHA_RE = re.compile(r"alpha_(m\d+|p\d+)\.png$")


def face_lower_roi(landmarks, width: int, height: int) -> np.ndarray:
    """Rasterize the existing jaw/chin-anchored lower-face geometry."""
    def point(index: int) -> np.ndarray:
        lm = landmarks[index]
        return np.array([lm.x * width, lm.y * height], dtype=np.float32)

    left, right = point(234), point(454)
    chin, forehead = point(152), point(10)
    face_height = max(float(chin[1] - forehead[1]), 1.0)
    center_x = float((left[0] + right[0]) / 2)
    top_y = float(np.clip((left[1] + right[1]) / 2 - .03 * face_height, 0, height - 1))
    bottom_y = min(float(chin[1] + .42 * face_height), height - 1)
    polygon = [
        (float(left[0]), top_y), (float(right[0]), top_y),
        (center_x + .70 * (right[0] - center_x), bottom_y),
        (center_x, bottom_y),
        (center_x - .70 * (center_x - left[0]), bottom_y),
    ]
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return np.asarray(mask) > 0


def load_landmarker(model_path: Path, deps_dir: Path):
    if str(deps_dir) not in sys.path:
        sys.path.append(str(deps_dir))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)), num_faces=1
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def make_color_overlay(rgb: np.ndarray, selected: np.ndarray) -> np.ndarray:
    color = np.zeros_like(rgb, dtype=np.uint8)
    color[selected] = (255, 60, 30)
    return np.uint8(np.clip(rgb.astype(np.float32) * .55 + color.astype(np.float32) * .45, 0, 255))


def resolve_images(args) -> list[Path]:
    if args.image:
        return [Path(value) for value in args.image]
    group_dir = args.images / args.category
    candidates = []
    for seed in args.seeds:
        for alpha in args.alphas:
            code = f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"
            path = group_dir / f"seed{seed:03d}_alpha_{code}.png"
            if path.exists():
                candidates.append(path)
            else:
                raise FileNotFoundError(path)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="feature images directory")
    parser.add_argument("--category", default="indian_man")
    parser.add_argument("--seeds", nargs="+", type=int, default=[30, 31])
    parser.add_argument("--alphas", nargs="+", type=int, default=[-7, 1, 7])
    parser.add_argument("--image", nargs="+", help="explicit image paths; overrides --category/--seeds/--alphas")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--onnx-model", type=Path, help="local ONNX file; downloads the ATR checkpoint if omitted")
    parser.add_argument("--mp-model", type=Path, default=DEFAULT_MP_MODEL)
    parser.add_argument("--mp-deps", type=Path, default=DEFAULT_MP_DEPS)
    args = parser.parse_args()

    import onnxruntime as ort

    onnx_path = args.onnx_model or (DEFAULT_MODEL_CACHE / "schp-atr-18-int8-static.onnx")
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading quantized SCHP ATR checkpoint to {onnx_path}", flush=True)
        temporary = onnx_path.with_suffix(onnx_path.suffix + ".part")
        urllib.request.urlretrieve(MODEL_URL, temporary)
        temporary.replace(onnx_path)
    print(f"Loading {onnx_path} for ONNX CPU inference", flush=True)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])
    mp, landmarker = load_landmarker(args.mp_model, args.mp_deps)
    rows = []
    try:
        for path in resolve_images(args):
            rgb_image = Image.open(path).convert("RGB")
            rgb = np.asarray(rgb_image)
            height, width = rgb.shape[:2]
            result = landmarker.detect(mp.Image.create_from_file(str(path)))
            if not result.face_landmarks:
                print(f"SKIP (no face landmarks): {path}", flush=True)
                continue
            lower_roi = face_lower_roi(result.face_landmarks[0], width, height)
            # SCHP's published processor resizes RGB to 512x512, scales to
            # [0,1], then applies the documented channel-wise normalization.
            resized = rgb_image.resize((512, 512), Image.Resampling.BILINEAR)
            values = np.asarray(resized, dtype=np.float32) / 255.0
            mean = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
            std = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)
            tensor = ((values - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
            logits = session.run(["logits"], {"pixel_values": tensor})[0][0]
            # Resize each class score map before argmax for a smooth boundary.
            full_size_logits = np.stack([
                np.asarray(Image.fromarray(class_logits.astype(np.float32)).resize(
                    (width, height), Image.Resampling.BILINEAR
                ))
                for class_logits in logits
            ])
            labels = np.argmax(full_size_logits, axis=0).astype(np.uint8)

            hair = labels == HAIR_CLASS_ID
            beard_candidate = hair & lower_roi
            out_dir = args.out / path.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = path.stem
            Image.fromarray(labels).save(out_dir / f"{stem}_class_ids.png")
            Image.fromarray(np.uint8(hair) * 255).save(out_dir / f"{stem}_hair_class_mask.png")
            Image.fromarray(np.uint8(lower_roi) * 255).save(out_dir / f"{stem}_lower_face_roi.png")
            Image.fromarray(np.uint8(beard_candidate) * 255).save(out_dir / f"{stem}_hair_in_lower_face_candidate.png")
            Image.fromarray(make_color_overlay(rgb, beard_candidate)).save(out_dir / f"{stem}_candidate_overlay.png")
            rows.append({
                "image": str(path.relative_to(ROOT)), "category": path.parent.name,
                "alpha": (ALPHA_RE.search(path.name).group(1) if ALPHA_RE.search(path.name) else ""),
                "hair_class_pixels_total": int(hair.sum()),
                "lower_face_roi_pixels": int(lower_roi.sum()),
                "hair_class_pixels_in_lower_face": int(beard_candidate.sum()),
                "candidate_pct_of_lower_face_roi": float(beard_candidate.sum() / max(lower_roi.sum(), 1) * 100),
            })
            print(f"Saved {path.name}: {beard_candidate.sum()} hair-class pixels in lower-face ROI", flush=True)
    finally:
        landmarker.close()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "probe_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["image"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
