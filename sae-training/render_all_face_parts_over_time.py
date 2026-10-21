"""Render every held-out portrait as part-by-checkpoint SAE heatmap grids."""
import argparse
import csv
import json
import re
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
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS, face_boxes, read_csv
from render_face_region_heatmaps import part_for_label

PARTS = ("face", "eyes", "eyebrows", "nose", "ears")
EPOCHS = (0, 4, 8, 12, 16, 20)
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")
MIN_CELLS = 2
MAX_PART_AREA = 0.12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "sae-training/runs/face_parts_sae_001")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--images", type=Path, default=ROOT / "backprop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_boxes(path):
    result = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        label = row["label"].strip().lower()
        if row["source"] == "objects" and label == "human face":
            region = "face"
        else:
            region = part_for_label(label)
        if region not in PARTS:
            continue
        coords = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        result[row["sample_id"]][region].append({"xyxy": coords, "label": row["label"]})
    return result


def category(row):
    return f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"


def box_mask(boxes, height, width, device="cpu"):
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (512 / width)
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (512 / height)
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        area_limit = 1.0 if item.get("part") == "face" else MAX_PART_AREA
        if ((x2 - x1) * (y2 - y1) / (512 * 512)) > area_limit:
            continue
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def masks_for(sample_id, shape, part_boxes, face_map, device="cpu"):
    height, width = shape[:2]
    face_box = [coord * 512 for coord in face_map[sample_id]]
    face = box_mask([{"xyxy": tuple(face_box), "part": "face"}], height, width, device)
    result = {}
    for part in PARTS:
        if part == "face":
            region = face
            reference = ~face
        else:
            items = [{**item, "part": part} for item in part_boxes.get(sample_id, {}).get(part, [])]
            region = box_mask(items, height, width, device)
            if part != "ears":
                region &= face
            reference = torch.ones_like(face) if part == "ears" else face
            reference &= ~region
        if int(region.sum()) < MIN_CELLS or int(reference.sum()) < MIN_CELLS:
            result[part] = (None, None)
        else:
            result[part] = (region, reference)
    return result


def validation_feature_selection(rows, boxes, face_map, residual_dir, checkpoint, device):
    model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
    groups = {part: defaultdict(list) for part in PARTS}
    with torch.no_grad():
        for row in rows:
            if row["split"] != "validation":
                continue
            sid = row["sample_id"]
            residual_path = residual_dir / f"{sid}.npy"
            if sid not in face_map or not residual_path.is_file():
                continue
            residual_np = np.load(residual_path, allow_pickle=False)
            residual = torch.from_numpy(residual_np.astype(np.float32)).to(device)
            codes = model.encode(residual)
            masks = masks_for(sid, residual.shape, boxes, face_map, device)
            for part in PARTS:
                region, reference = masks[part]
                if region is not None:
                    groups[part][category(row)].append(
                        (codes[region].mean(dim=0).cpu().numpy()
                         - codes[reference].mean(dim=0).cpu().numpy())
                    )
            del residual, codes
    selected, scores, counts = {}, {}, {}
    for part in PARTS:
        by_category = {key: np.mean(items, axis=0) for key, items in groups[part].items()}
        if not by_category:
            raise ValueError(f"No usable validation boxes for {part}")
        macro = np.mean(list(by_category.values()), axis=0)
        feature = int(np.argmax(macro))
        selected[part] = feature
        scores[part] = float(macro[feature])
        counts[part] = {"images": sum(len(items) for items in groups[part].values()),
                        "categories": len(groups[part])}
    del model
    return selected, scores, counts


def collect_test_maps(rows, feature_by_part, run_dir, residual_dir, face_map, device):
    test_rows = [row for row in rows if row["split"] == "test"
                 and row["sample_id"] in face_map
                 and (residual_dir / f"{row['sample_id']}.npy").is_file()]
    epochs = {}
    for epoch in EPOCHS:
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:03d}"
        model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
        model.requires_grad_(False)
        maps = {}
        with torch.no_grad():
            for index, row in enumerate(test_rows, 1):
                sid = row["sample_id"]
                array = np.load(residual_dir / f"{sid}.npy", allow_pickle=False)
                residual = torch.from_numpy(array.astype(np.float32)).to(device)
                codes = model.encode(residual)
                maps[sid] = {part: codes[..., feature].float().cpu().numpy()
                             for part, feature in feature_by_part.items()}
                del residual, codes
                if index % 60 == 0:
                    print(f"epoch={epoch} encoded {index}/{len(test_rows)}", flush=True)
        epochs[epoch] = maps
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return test_rows, epochs


def overlay(image, values, face_box_norm, target_boxes, part):
    face_box = [coord * 512 for coord in face_box_norm]
    height, width = values.shape
    face = box_mask([{"xyxy": tuple(face_box), "part": "face"}], height, width).numpy()
    region_values = values[face]
    vmax = max(float(region_values.max()) if region_values.size else 0.0, 1e-8)
    normalized = np.clip(values / vmax, 0, 1) * face
    red = np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1)
    rgba = np.uint8(np.stack((red, green, blue, np.ones_like(normalized)), axis=-1) * 255)
    rgba[..., 3] = np.uint8(normalized * 170)
    heat = Image.fromarray(rgba, mode="RGBA").resize(image.size, Image.Resampling.NEAREST)
    result = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")
    draw = ImageDraw.Draw(result)
    sx, sy = image.width / 512, image.height / 512
    x1, y1, x2, y2 = face_box
    draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline="#00e5ff", width=2)
    for item in target_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline="#ffff00", width=2)
    return result


def render_individual(row, boxes, face_map, epoch_maps, features, images_dir, output_dir):
    sid = row["sample_id"]
    category_name = category(row)
    tile = 128
    label_width, header_h = 148, 34
    sheet = Image.new("RGB", (label_width + tile * len(EPOCHS), header_h + tile * len(PARTS)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for epoch_index, epoch in enumerate(EPOCHS):
        draw.text((label_width + epoch_index * tile + 42, 10), f"epoch {epoch}",
                  fill=(25, 40, 55), font=font)
    image_path = images_dir / row["image"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    for part_index, part in enumerate(PARTS):
        items = boxes.get(sid, {}).get(part, []) if part != "face" else []
        part_exists = part == "face" or bool(items)
        row_label = f"{part.title()} · f{features[part]}" + ("" if part_exists else " · no box")
        draw.text((5, header_h + part_index * tile + 52), row_label, fill=(25, 40, 55), font=font)
        for epoch_index, epoch in enumerate(EPOCHS):
            values = epoch_maps[epoch][sid][part]
            rendered = overlay(image, values, face_map[sid], items, part)
            rendered = rendered.resize((tile, tile), Image.Resampling.LANCZOS)
            x = label_width + epoch_index * tile
            y = header_h + part_index * tile
            sheet.paste(rendered, (x, y))
            if not part_exists:
                ImageDraw.Draw(sheet).text((x + 5, y + 5), "NO PART BOX", fill="#fff200", font=font,
                                           stroke_width=1, stroke_fill="#222222")
    out = output_dir / "sheets" / f"{sid}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=86, optimize=True)
    return out


def write_gallery(rows, boxes, output_dir):
    by_category = defaultdict(list)
    for row in rows:
        by_category[category(row)].append(row)
    markdown = [
        "# Held-out face-part feature maps",
        "",
        f"All {len(rows)} held-out portraits are shown below. Each sheet has rows for face, eyes, eyebrows, nose, and ears, and columns for epochs 0, 4, 8, 12, 16, and 20.",
        "",
        "Cyan outlines the detected face box; yellow outlines the target part box. A missing box means there is no usable automatic annotation, not that the part is absent. Feature IDs were selected on validation data. See the [validation feature table](validation_selected_features.csv) and [full report](../REPORT.md).",
        "",
    ]
    for category_name in sorted(by_category):
        markdown.extend([f"## {category_name.replace('_', ' ').title()}", ""])
        for row in sorted(by_category[category_name], key=lambda item: int(item["seed"])):
            sid = row["sample_id"]
            boxes_count = sum(bool(boxes.get(sid, {}).get(part)) for part in PARTS if part != "face")
            markdown.extend([
                f"### {sid} · {boxes_count} part types with boxes",
                "",
                f"[![{sid}: face, eyes, eyebrows, nose, and ears across six epochs](sheets/{sid}.jpg)](sheets/{sid}.jpg)",
                "",
            ])
    (output_dir / "index.md").write_text("\n".join(markdown) + "\n")


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    boxes = load_boxes(args.boxes)
    face_map = face_boxes(args.boxes)
    device = torch.device(args.device)
    feature_by_part, val_scores, val_counts = validation_feature_selection(
        rows, boxes, face_map, args.residuals, args.run / "checkpoints/epoch_020", device
    )
    test_rows, epoch_maps = collect_test_maps(rows, feature_by_part, args.run,
                                              args.residuals, face_map, device)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(test_rows, 1):
        render_individual(row, boxes, face_map, epoch_maps, feature_by_part,
                          args.images, args.output)
        if index % 40 == 0:
            print(f"rendered {index}/{len(test_rows)} individual grids", flush=True)
    write_gallery(test_rows, boxes, args.output)
    with (args.output / "validation_selected_features.csv").open("w", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("part", "feature", "validation_macro_contrast", "validation_images", "validation_categories"))
        for part in PARTS:
            writer.writerow((part, feature_by_part[part], val_scores[part],
                             val_counts[part]["images"], val_counts[part]["categories"]))
    (args.output / "settings.json").write_text(json.dumps({
        "run": str(args.run), "sample_split": "test", "test_images": len(test_rows),
        "parts": list(PARTS), "epochs": list(EPOCHS),
        "feature_selection": "one feature per part selected by macro-category activation contrast on validation data at epoch 20",
        "feature_by_part": feature_by_part,
        "validation_coverage": val_counts,
        "box_sources": str(args.boxes),
        "part_box_max_area_fraction": MAX_PART_AREA,
        "overlay": "jet heatmap normalized within detected face; cyan face box; yellow part box; nearest-neighbor activation upscaling",
        "missing_box_interpretation": "No usable automatic box is missing annotation evidence, not absence of the part.",
    }, indent=2) + "\n")
    print(f"Saved {len(test_rows)} per-image grids and gallery to {args.output}", flush=True)


if __name__ == "__main__":
    main()
