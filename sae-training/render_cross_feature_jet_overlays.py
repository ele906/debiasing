"""Render validation and test portraits with jet maps for every selected feature/region pair."""

import argparse
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
from render_all_face_parts_over_time import PARTS, category, load_boxes, masks_for, overlay

RUN = ROOT / "sae-training/runs/face_parts_sae_001"
VALIDATION_DATA = ROOT / "backprop"
VALIDATION_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
TEST_DATA = RUN / "independent_test_seed200_299"
TEST_BOXES = TEST_DATA / "annotations/bounding_boxes.csv"
FEATURES_CSV = RUN / "individual_part_epoch_grids/validation_selected_features.csv"
OUTPUT = TEST_DATA / "test_contrast/feature_cross_contrast"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--validation-data", type=Path, default=VALIDATION_DATA)
    parser.add_argument("--validation-boxes", type=Path, default=VALIDATION_BOXES)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--test-boxes", type=Path, default=TEST_BOXES)
    parser.add_argument("--features", type=Path, default=FEATURES_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT / "cross_feature_jet_overlays.jpg")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def selected_features(path):
    mapping = []
    seen = set()
    for row in read_csv(path):
        feature = int(row["feature"])
        if feature not in seen:
            mapping.append({"feature": feature, "selected_for": []})
            seen.add(feature)
        next(item for item in mapping if item["feature"] == feature)["selected_for"].append(row["part"])
    return mapping


def usable_by_region(data_root, boxes_path, split):
    rows = [row for row in read_csv(data_root / "manifest.csv") if row["split"] == split]
    boxes = load_boxes(boxes_path)
    faces = face_boxes(boxes_path)
    residuals = data_root / "residuals" if split == "test" else ROOT / "sae-training/residuals"
    usable = defaultdict(list)
    for row in rows:
        sid = row["sample_id"]
        residual_path = residuals / f"{sid}.npy"
        if sid not in faces or not residual_path.is_file():
            continue
        shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
        masks = masks_for(sid, shape, boxes, faces, "cpu")
        for part in PARTS:
            if masks[part][0] is not None:
                usable[part].append(row)
    chosen = {}
    for part in PARTS:
        ordered = sorted(usable[part], key=lambda row: (category(row), int(row["seed"])))
        if not ordered:
            raise ValueError(f"No usable {split} portrait for region {part}")
        chosen[part] = ordered[0]
    return chosen, boxes, faces, residuals


def render_split(model, split, data_root, boxes_path, features, device):
    split_name = "validation" if split == "validation" else "test"
    chosen, boxes, faces, residuals = usable_by_region(data_root, boxes_path, split_name)
    overlays = {}
    with torch.inference_mode():
        for part, row in chosen.items():
            sid = row["sample_id"]
            array = np.load(residuals / f"{sid}.npy", allow_pickle=False)
            residual = torch.from_numpy(array.astype(np.float32, copy=False)).to(device)
            codes = model.encode(residual)
            image_path = data_root / row["image"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            target_boxes = boxes.get(sid, {}).get(part, []) if part != "face" else []
            for item in features:
                feature = item["feature"]
                values = codes[..., feature].float().cpu().numpy()
                overlays[part, feature] = overlay(image, values, faces[sid], target_boxes, part)
            print(f"{split_name}: example {part}: {sid}", flush=True)
            del codes, residual
    return overlays, chosen


def make_figure(validation, validation_rows, test, test_rows, features, output):
    row_header, tile, label_h, split_h = 180, 210, 34, 36
    panel_w = len(PARTS) * tile
    width = row_header + 2 * panel_w
    height = split_h + label_h + len(features) * tile
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    validation_x = row_header
    test_x = row_header + panel_w
    draw.text((validation_x + panel_w // 2 - 30, 10), "VALIDATION", fill="#111111", font=font)
    draw.text((test_x + panel_w // 2 - 45, 10), "INDEPENDENT TEST", fill="#111111", font=font)
    for col, part in enumerate(PARTS):
        draw.text((validation_x + col * tile + 70, split_h + 8), part.title(), fill="#111111", font=font)
        draw.text((test_x + col * tile + 70, split_h + 8), part.title(), fill="#111111", font=font)
    for row_index, item in enumerate(features):
        y = split_h + label_h + row_index * tile
        feature = item["feature"]
        selected_for = "/".join(item["selected_for"])
        draw.text((8, y + 60), f"SAE feature {feature}\nselected for {selected_for}",
                  fill="#111111", font=font, spacing=5)
        for col, part in enumerate(PARTS):
            for x0, overlays, chosen in ((validation_x, validation, validation_rows),
                                         (test_x, test, test_rows)):
                x = x0 + col * tile
                rendered = overlays[part, feature].copy()
                rendered.thumbnail((tile - 8, tile - label_h - 4), Image.Resampling.LANCZOS)
                sheet.paste(rendered, (x + (tile - rendered.width) // 2, y + label_h))
                row = chosen[part]
                note = f"{category(row).replace('_', ' ')} · {row['seed']}"
                draw.text((x + 4, y + 5), note[:28], fill="#111111", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=91, optimize=True)


def make_feature_contact_sheets(validation, validation_rows, test, test_rows, features, output_dir):
    """Write one readable contact sheet per feature, with validation/test rows."""
    cell_w, cell_h = 250, 285
    title_h, split_label_w = 48, 92
    width = split_label_w + len(PARTS) * cell_w
    height = title_h + 2 * cell_h
    font = ImageFont.load_default()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for item in features:
        feature = item["feature"]
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"Feature {feature} · selected for {' / '.join(item['selected_for'])} · jet overlays",
                  fill="#111111", font=font)
        for row_index, (split_name, overlays, chosen) in enumerate((
                ("Validation", validation, validation_rows),
                ("Independent test", test, test_rows))):
            y = title_h + row_index * cell_h
            draw.text((7, y + cell_h // 2 - 10), split_name, fill="#111111", font=font)
            for col, part in enumerate(PARTS):
                x = split_label_w + col * cell_w
                row = chosen[part]
                draw.text((x + 5, y + 5), part.title(), fill="#111111", font=font)
                draw.text((x + 5, y + 20),
                          f"{category(row).replace('_', ' ').title()} · {row['seed']}"[:34],
                          fill="#333333", font=font)
                rendered = overlays[part, feature].copy()
                rendered.thumbnail((cell_w - 10, cell_h - 50), Image.Resampling.LANCZOS)
                sheet.paste(rendered, (x + (cell_w - rendered.width) // 2, y + 42))
        path = output_dir / f"feature_{feature}_contact_sheet.jpg"
        sheet.save(path, quality=91, optimize=True)
        results.append(str(path))
    return results


def main():
    args = parse_args()
    run = args.run.resolve()
    features = selected_features(args.features.resolve())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = run / "checkpoints/epoch_020"
    print(f"Loading epoch-20 SAE from {checkpoint} on {device}", flush=True)
    model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
    validation, validation_rows = render_split(
        model, "validation", args.validation_data.resolve(), args.validation_boxes.resolve(), features, device
    )
    test, test_rows = render_split(
        model, "test", args.test_data.resolve(), args.test_boxes.resolve(), features, device
    )
    output = args.output.resolve()
    make_figure(validation, validation_rows, test, test_rows, features, output)
    contact_sheets = make_feature_contact_sheets(
        validation, validation_rows, test, test_rows, features, output.parent / "contact_sheets"
    )
    metadata = {
        "features": features,
        "regions": list(PARTS),
        "validation_examples": {part: row["sample_id"] for part, row in validation_rows.items()},
        "test_examples": {part: row["sample_id"] for part, row in test_rows.items()},
        "per_feature_contact_sheets": contact_sheets,
        "checkpoint": str(checkpoint),
        "overlay": "jet feature activation normalized within detected face; cyan outline is face; yellow outline is target region when distinct from face",
        "note": "One representative image is shown per split/region; these examples illustrate maps and are not used to calculate contrast means or intervals.",
    }
    output.with_name("cross_feature_jet_overlays.settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved jet overlay comparison to {output}", flush=True)


if __name__ == "__main__":
    main()
