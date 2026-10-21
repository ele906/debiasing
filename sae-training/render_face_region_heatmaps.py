"""Compare SAE feature activations inside annotated facial regions."""
import argparse
import csv
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "backprop"
DEFAULT_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
DEFAULT_RESIDUALS = ROOT / "sae-training/residuals"
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")
REGIONS = ("face", "eyes", "eyebrows", "nose", "mouth", "ears")
BLOCK = "unet.down_blocks.2.attentions.1"
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="SAE run with checkpoints/ and feature ranking")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--output", type=Path, help="Output folder (default: RUN/region_heatmaps)")
    parser.add_argument("--feature-count", type=int, default=3,
                        help="Number of top validation-ranked features to render")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Overwrite sheets in an existing output folder")
    args = parser.parse_args()
    if args.feature_count < 1:
        parser.error("--feature-count must be positive")
    args.run = args.run.resolve()
    args.data = args.data.resolve()
    args.boxes = args.boxes.resolve()
    args.residuals = args.residuals.resolve()
    args.output = (args.output or args.run / "region_heatmaps").resolve()
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output already exists; choose a new --output: {args.output}")
    return args


def load_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def part_for_label(label):
    text = label.lower().strip()
    if re.search(r"\bearrings?\b", text):
        return None
    if re.search(r"\beyebrows?\b", text):
        return "eyebrows"
    if re.search(r"\beyes?\b", text):
        return "eyes"
    if re.search(r"\bnose\b", text):
        return "nose"
    if re.search(r"\b(?:mouth|lips?)\b", text):
        return "mouth"
    if re.search(r"\bear\b|\bears\b", text):
        return "ears"
    if re.search(r"\bface\b", text):
        return "face"
    return None


def load_region_boxes(path):
    boxes = defaultdict(lambda: defaultdict(list))
    for row in load_rows(path):
        part = part_for_label(row["label"])
        if part is None:
            continue
        coordinates = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        boxes[row["sample_id"]][part].append({"xyxy": coordinates, "source": row["source"],
                                               "label": row["label"]})
    return boxes


def choose_samples(data_dir, boxes_by_sample, residual_dir):
    rows = [row for row in load_rows(data_dir / "manifest.csv") if row["split"] == "test"]
    selected = {}
    for race in RACES:
        for gender in GENDERS:
            category = (race, gender)
            category_rows = sorted(
                (row for row in rows
                 if row["race_prompt"].replace(" ", "_") == race
                 and row["gender_prompt"] == gender),
                key=lambda row: int(row["seed"]),
            )
            for region in REGIONS:
                sample = next((row for row in category_rows
                               if region in boxes_by_sample.get(row["sample_id"], {})
                               and (residual_dir / f"{row['sample_id']}.npy").is_file()), None)
                selected[region, category] = sample
    return selected


def load_features(run_dir, count):
    region_ranking = run_dir / "region_localized_features.csv"
    if region_ranking.is_file():
        by_region = {region: [] for region in REGIONS}
        with region_ranking.open(newline="") as source:
            for row in csv.DictReader(source):
                region = row["region"]
                if region in by_region and len(by_region[region]) < count:
                    by_region[region].append(int(row["feature"]))
        return by_region
    ranking_path = run_dir / "eye_localized_features.csv"
    if not ranking_path.is_file():
        ranking_path = run_dir / "face_localized_features.csv"
    with ranking_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    features = [int(row["feature"]) for row in rows[:count]]
    if not features:
        raise ValueError(f"No ranked features in {ranking_path}")
    return {region: features for region in REGIONS}


def load_snapshots(run_dir):
    snapshots = sorted((run_dir / "checkpoints").glob("epoch_[0-9][0-9][0-9]"))
    if not snapshots or snapshots[0].name != "epoch_000":
        raise ValueError("Expected checkpoints/epoch_000 and later epoch snapshots")
    return snapshots


def region_mask(boxes, height, width, device):
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (512 / width)
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (512 / height)
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def collect_activations(selected, boxes_by_sample, residual_dir, snapshots, features, device):
    sample_ids = sorted({row["sample_id"] for row in selected.values() if row is not None})
    activations = {}
    for snapshot in snapshots:
        sae = SparseAutoencoder.load_from_disk(str(snapshot)).float().to(device).eval()
        sae.requires_grad_(False)
        epoch = int(snapshot.name.split("_")[1])
        for sample_id in sample_ids:
            residual_np = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
            residual = torch.from_numpy(residual_np.astype(np.float32)).to(device)
            with torch.no_grad():
                codes = sae.encode(residual)
            for feature in features:
                activations[epoch, sample_id, feature] = codes[..., feature].float().cpu().numpy()
            del residual, codes
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return activations


def make_overlay(image, values, mask, boxes):
    region_values = values[mask]
    vmax = float(region_values.max()) if region_values.size else 0.0
    normalized = np.clip(values / max(vmax, 1e-8), 0, 1) * mask
    red = np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1)
    rgb = np.uint8(np.stack((red, green, blue), axis=-1) * 255)
    alpha = np.uint8(normalized * 180)
    heatmap = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
    heatmap = heatmap.resize(image.size, resample=Image.Resampling.NEAREST)
    result = Image.alpha_composite(image.convert("RGBA"), heatmap).convert("RGB")
    draw = ImageDraw.Draw(result)
    scale_x, scale_y = image.width / 512, image.height / 512
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        draw.rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                       outline="#00e5ff", width=2)
    return result


def save_sheets(selected, boxes_by_sample, activations, epochs, features_by_region, data_dir, output_dir):
    tile_size, label_width, header_height, row_height = 160, 180, 44, 170
    categories = [(race, gender) for race in RACES for gender in GENDERS]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for region in REGIONS:
        for feature in features_by_region[region]:
            sheet = Image.new("RGB", (label_width + len(epochs) * tile_size,
                                       header_height + len(categories) * row_height), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 4), f"{region.title()} · feature {feature}", fill="black")
            for column, epoch in enumerate(epochs):
                draw.text((label_width + column * tile_size + 5, 22), f"epoch {epoch}", fill="black")
            for row_index, category in enumerate(categories):
                y = header_height + row_index * row_height
                race, gender = category
                draw.text((4, y + 7), f"{race.replace('_', ' ').title()} {gender.title()}", fill="black")
                row = selected[region, category]
                if row is None:
                    draw.text((4, y + 28), "no part box", fill="gray")
                    continue
                sample_id = row["sample_id"]
                draw.text((4, y + 28), f"seed {row['seed']}", fill="gray")
                with Image.open(data_dir / row["image"]) as source:
                    image = source.convert("RGB")
                boxes = boxes_by_sample[sample_id][region]
                mask = region_mask(boxes, *activations[epochs[0], sample_id, feature].shape,
                                   torch.device("cpu")).numpy()
                for column, epoch in enumerate(epochs):
                    feature_map = activations[epoch, sample_id, feature]
                    selected_values = feature_map[mask]
                    summary_rows.append({
                        "region": region,
                        "race": race,
                        "gender": gender,
                        "sample_id": sample_id,
                        "feature": feature,
                        "epoch": epoch,
                        "mean_activation_in_region": float(selected_values.mean()) if selected_values.size else 0.0,
                        "max_activation_in_region": float(selected_values.max()) if selected_values.size else 0.0,
                        "active_cell_fraction": float((selected_values > 0).mean()) if selected_values.size else 0.0,
                    })
                    overlay = make_overlay(image, feature_map, mask, boxes)
                    tile = overlay.resize((tile_size - 8, tile_size - 8))
                    sheet.paste(tile, (label_width + column * tile_size + 4, y + 4))
            region_dir = output_dir / region
            region_dir.mkdir(exist_ok=True)
            sheet.save(region_dir / f"feature_{feature}.jpg", quality=92)
    if summary_rows:
        with (output_dir / "regional_activation_summary.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(summary_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary_rows)


def main():
    args = parse_args()
    settings = json.loads((args.run / "settings.json").read_text())
    if settings.get("block") != BLOCK:
        raise ValueError(f"Expected SAE block {BLOCK}; found {settings.get('block')}")
    boxes_by_sample = load_region_boxes(args.boxes)
    selected = choose_samples(args.data, boxes_by_sample, args.residuals)
    features_by_region = load_features(args.run, args.feature_count)
    features = sorted({feature for values in features_by_region.values() for feature in values})
    if not features:
        raise ValueError("No ranked features were found for the requested regions")
    snapshots = load_snapshots(args.run)
    device = torch.device(args.device)
    activations = collect_activations(selected, boxes_by_sample, args.residuals,
                                      snapshots, features, device)
    save_sheets(selected, boxes_by_sample, activations,
                [int(snapshot.name.split("_")[1]) for snapshot in snapshots],
                features_by_region, args.data, args.output)
    summary = {
        "epochs": [int(snapshot.name.split("_")[1]) for snapshot in snapshots],
        "features_by_region": features_by_region,
        "regions": list(REGIONS),
        "sample_ids": {
            f"{region}/{race}_{gender}": row["sample_id"] if row else None
            for (region, (race, gender)), row in selected.items()
        },
        "annotation_source": str(args.boxes),
        "annotation_note": "Automatic boxes; earrings are excluded from ear regions; inspect overlays before interpreting.",
    }
    (args.output / "settings.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved region-specific feature sheets to {args.output}")


if __name__ == "__main__":
    main()
