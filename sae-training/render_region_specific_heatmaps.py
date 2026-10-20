"""Render each facial region's own validation-ranked SAE features over epochs."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS
from render_face_region_heatmaps import GENDERS, RACES, load_region_boxes, make_overlay, region_mask

REGIONS = ("eyes", "eyebrows", "nose", "ears")
MIN_CELLS = 2
MAX_BOX_AREA = 0.12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Fine-tuning run with epoch checkpoints and part validation")
    parser.add_argument("--data", type=Path, default=DEFAULT_MANIFEST.parent)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--output", type=Path, help="Default: RUN/region_specific_heatmaps")
    parser.add_argument("--feature-count", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Replace sheets in an existing output folder")
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.data = args.data.resolve()
    args.boxes = args.boxes.resolve()
    args.residuals = args.residuals.resolve()
    args.output = (args.output or args.run / "region_specific_heatmaps").resolve()
    if args.feature_count < 1:
        parser.error("--feature-count must be positive")
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output already exists; choose a new --output or use --resume: {args.output}")
    return args


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def load_region_features(run_dir, count):
    path = run_dir / "part_feature_validation" / "heldout_feature_validation.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run held-out part validation first; missing {path}")
    by_region = defaultdict(list)
    for row in read_rows(path):
        if row["region"] in REGIONS:
            feature = int(row["feature"])
            if feature not in by_region[row["region"]]:
                by_region[row["region"]].append(feature)
    missing = [region for region in REGIONS if len(by_region[region]) < count]
    if missing:
        raise ValueError(f"Not enough validation-ranked features for: {', '.join(missing)}")
    return {region: by_region[region][:count] for region in REGIONS}


def valid_boxes(boxes, height, width):
    accepted = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        area_fraction = (x2 - x1) * (y2 - y1) / (512 * 512)
        if area_fraction <= MAX_BOX_AREA:
            accepted.append(box)
    if not accepted:
        return []
    mask = region_mask(accepted, height, width, torch.device("cpu"))
    return accepted if int(mask.sum()) >= MIN_CELLS else []


def select_test_samples(manifest_path, region_boxes, residual_dir):
    rows = [row for row in read_rows(manifest_path) if row["split"] == "test"]
    selected = {}
    selected_boxes = {}
    for race in RACES:
        for gender in GENDERS:
            category_rows = sorted(
                (row for row in rows
                 if row["race_prompt"].replace(" ", "_") == race
                 and row["gender_prompt"] == gender),
                key=lambda row: int(row["seed"]),
            )
            for region in REGIONS:
                selected[region, race, gender] = None
                for row in category_rows:
                    sample_id = row["sample_id"]
                    boxes = region_boxes.get(sample_id, {}).get(region, [])
                    if not boxes:
                        continue
                    residual_path = residual_dir / f"{sample_id}.npy"
                    if not residual_path.is_file():
                        continue
                    shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
                    accepted = valid_boxes(boxes, *shape[:2])
                    if accepted:
                        selected[region, race, gender] = row
                        selected_boxes[sample_id, region] = accepted
                        break
    return selected, selected_boxes


def collect_activations(selected, snapshots, features_by_region, residual_dir, device):
    sample_ids = sorted({row["sample_id"] for row in selected.values() if row is not None})
    all_features = sorted({feature for features in features_by_region.values() for feature in features})
    activations = {}
    for snapshot in snapshots:
        epoch = int(snapshot.name.split("_")[1])
        sae = SparseAutoencoder.load_from_disk(str(snapshot)).float().to(device).eval()
        sae.requires_grad_(False)
        for sample_id in sample_ids:
            array = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
            residual = torch.from_numpy(array.astype(np.float32)).to(device)
            with torch.no_grad():
                codes = sae.encode(residual)
            for feature in all_features:
                activations[epoch, sample_id, feature] = codes[..., feature].float().cpu().numpy()
            del residual, codes
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Encoded epoch {epoch:03d} for {len(sample_ids)} held-out images", flush=True)
    return activations


def save_sheets(selected, selected_boxes, activations, features_by_region, epochs, manifest_dir, output):
    categories = [(race, gender) for race in RACES for gender in GENDERS]
    tile_size, label_width, header_height, row_height = 160, 180, 44, 170
    output.mkdir(parents=True, exist_ok=True)
    for region, features in features_by_region.items():
        region_dir = output / region
        region_dir.mkdir(parents=True, exist_ok=True)
        for feature in features:
            sheet = Image.new("RGB", (label_width + len(epochs) * tile_size,
                                       header_height + len(categories) * row_height), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 4), f"{region.title()} · feature {feature}", fill="black")
            for column, epoch in enumerate(epochs):
                draw.text((label_width + column * tile_size + 5, 22), f"epoch {epoch}", fill="black")
            for row_index, (race, gender) in enumerate(categories):
                y = header_height + row_index * row_height
                draw.text((4, y + 7), f"{race.replace('_', ' ').title()} {gender.title()}", fill="black")
                row = selected[region, race, gender]
                if row is None:
                    draw.text((4, y + 28), "no accepted region box", fill="gray")
                    continue
                sample_id = row["sample_id"]
                draw.text((4, y + 28), f"seed {row['seed']}", fill="gray")
                with Image.open(manifest_dir / row["image"]) as source:
                    image = source.convert("RGB")
                boxes = selected_boxes[sample_id, region]
                first_map = activations[epochs[0], sample_id, feature]
                mask = region_mask(boxes, *first_map.shape, torch.device("cpu")).numpy()
                for column, epoch in enumerate(epochs):
                    feature_map = activations[epoch, sample_id, feature]
                    overlay = make_overlay(image, feature_map, mask, boxes)
                    tile = overlay.resize((tile_size - 8, tile_size - 8))
                    sheet.paste(tile, (label_width + column * tile_size + 4, y + 4))
            sheet.save(region_dir / f"feature_{feature}.jpg", quality=92)


def main():
    args = parse_args()
    validation_path = args.run / "part_feature_validation" / "heldout_feature_validation.csv"
    features_by_region = load_region_features(args.run, args.feature_count)
    region_boxes = load_region_boxes(args.boxes)
    selected, selected_boxes = select_test_samples(args.data / "manifest.csv", region_boxes,
                                                   args.residuals)
    snapshots = sorted((args.run / "checkpoints").glob("epoch_[0-9][0-9][0-9]"))
    if not snapshots or snapshots[0].name != "epoch_000":
        raise ValueError("Expected checkpoints/epoch_000 and later snapshots")
    epochs = [int(snapshot.name.split("_")[1]) for snapshot in snapshots]
    sample_count = sum(row is not None for row in selected.values())
    if sample_count == 0:
        raise ValueError("No held-out samples have usable boxes and residual activations")
    device = torch.device(args.device)
    activations = collect_activations(selected, snapshots, features_by_region,
                                      args.residuals, device)
    save_sheets(selected, selected_boxes, activations, features_by_region, epochs,
                args.data, args.output)
    metadata = {
        "features_by_region": features_by_region,
        "feature_selection": "Highest validation macro-category activation contrast from part_feature_validation/heldout_feature_validation.csv",
        "sample_split": "test",
        "epochs": epochs,
        "images_with_region_boxes": {
            f"{region}/{race}_{gender}": row["sample_id"] if row else None
            for (region, race, gender), row in selected.items()
        },
        "box_filter": {"maximum_area_fraction": MAX_BOX_AREA,
                       "minimum_activation_cells": MIN_CELLS},
        "overlay": "Nearest-neighbor pixelated jet activation, shown only inside the selected region box",
        "validation_source": str(validation_path),
        "interpretation": "Region candidates were selected on validation data and visualized on held-out test images; boxes are automatic and unreviewed.",
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved region-specific epoch heatmaps to {args.output}", flush=True)


if __name__ == "__main__":
    main()
