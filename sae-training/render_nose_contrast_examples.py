"""Render nose feature 1664 overlays on validation and independent test portraits."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from fine_tune_face_sae import face_boxes, read_csv
from render_all_face_parts_over_time import category, load_boxes, masks_for, overlay

RUN = ROOT / "sae-training/runs/face_parts_sae_001"
VALIDATION_ROOT = ROOT / "backprop"
TEST_ROOT = RUN / "independent_test_seed200_299"
VALIDATION_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
TEST_BOXES = TEST_ROOT / "annotations/bounding_boxes.csv"
FEATURE = 1664
PART = "nose"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--validation", type=Path, default=VALIDATION_ROOT)
    parser.add_argument("--validation-boxes", type=Path, default=VALIDATION_BOXES)
    parser.add_argument("--test", type=Path, default=TEST_ROOT)
    parser.add_argument("--test-boxes", type=Path, default=TEST_BOXES)
    parser.add_argument("--output", type=Path, default=TEST_ROOT / "test_contrast/nose_examples")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    return args


def load_split(data_root, split):
    rows = [row for row in read_csv(data_root / "manifest.csv") if row["split"] == split]
    return rows


def select_rows(rows, boxes, face_map, residual_dir, per_category):
    candidates = defaultdict(list)
    for row in rows:
        sid = row["sample_id"]
        residual_path = residual_dir / f"{sid}.npy"
        if sid not in face_map or not residual_path.is_file():
            continue
        shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
        region, reference = masks_for(sid, shape, boxes, face_map, "cpu")[PART]
        if region is None:
            continue
        candidates[category(row)].append(row)
    selected = []
    for cat in sorted(candidates):
        group = sorted(candidates[cat], key=lambda r: int(r["seed"]))
        selected.extend(group[:per_category])
    return selected


def contact_sheet(items, output, title, columns=4):
    cell_w, cell_h, title_h = 280, 310, 42
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="#111111", font=font)
    for index, (row, image_path) in enumerate(items):
        r, c = divmod(index, columns)
        x, y = c * cell_w, title_h + r * cell_h
        label = f"{category(row).replace('_', ' ').title()} · seed {row['seed']}"
        draw.text((x + 6, y + 6), label, fill="#111111", font=font)
        with Image.open(image_path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((cell_w - 12, cell_h - 34), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x + (cell_w - tile.width) // 2, y + 27))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True)


def render_split(model, split_name, data_root, boxes_path, output_root, per_category, device):
    rows = load_split(data_root, "validation" if split_name == "validation" else "test")
    boxes = load_boxes(boxes_path)
    face_map = face_boxes(boxes_path)
    residual_dir = data_root / "residuals" if split_name == "test" else ROOT / "sae-training/residuals"
    image_root = data_root
    chosen = select_rows(rows, boxes, face_map, residual_dir, per_category)
    output_dir = output_root / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_items = []
    used_categories = set()
    with torch.inference_mode():
        for index, row in enumerate(chosen, 1):
            sid = row["sample_id"]
            residual_np = np.load(residual_dir / f"{sid}.npy", allow_pickle=False)
            residual = torch.from_numpy(residual_np.astype(np.float32, copy=False)).to(device)
            code = model.encode(residual)
            values = code[..., FEATURE].float().cpu().numpy()
            face_box = face_map[sid]
            nose_boxes = boxes.get(sid, {}).get(PART, [])
            image_path = image_root / row["image"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            rendered = overlay(image, values, face_box, nose_boxes, PART)
            out_path = output_dir / f"{sid}_f{FEATURE}_jet.jpg"
            rendered.save(out_path, quality=90, optimize=True)
            contact_items.append((row, out_path))
            used_categories.add(category(row))
            if index % 8 == 0:
                print(f"{split_name}: rendered {index}/{len(chosen)}", flush=True)
            del code, residual
    contact_sheet(contact_items, output_dir / "contact_sheet.jpg",
                  f"{split_name.title()} portraits · nose feature {FEATURE} · jet activation + boxes")
    return {"images": len(contact_items), "categories": len(used_categories),
            "examples": [row["sample_id"] for row, _ in contact_items]}


def main():
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    selected_file = run / "individual_part_epoch_grids/validation_selected_features.csv"
    with selected_file.open(newline="") as source:
        nose_row = next(row for row in csv.DictReader(source) if row["part"] == PART)
    if int(nose_row["feature"]) != FEATURE:
        raise ValueError(f"Expected validation-selected nose feature {FEATURE}, got {nose_row['feature']}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = run / "checkpoints/epoch_020"
    print(f"Loading SAE checkpoint {checkpoint} on {device}", flush=True)
    model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
    validation_info = render_split(model, "validation", args.validation.resolve(),
                                   args.validation_boxes.resolve(), output,
                                   args.per_category, device)
    test_info = render_split(model, "test", args.test.resolve(), args.test_boxes.resolve(),
                             output, args.per_category, device)
    settings = {
        "feature": FEATURE,
        "part": PART,
        "checkpoint": str(checkpoint),
        "feature_selection": "selected on validation data at epoch 20 by macro-category activation contrast",
        "validation_gallery": validation_info,
        "independent_test_gallery": test_info,
        "per_category_limit": args.per_category,
        "overlay": "jet feature activation normalized within detected face; cyan face box; yellow nose box",
        "annotations": "automatic region boxes, not human-reviewed",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved validation and test nose overlays to {output}", flush=True)


if __name__ == "__main__":
    main()
