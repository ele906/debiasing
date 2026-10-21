"""Render fixed pretrained-SAE feature maps over 25 annotated portraits/category."""
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
sys.path.insert(0, str(ROOT / "sae-training"))
from SAE import SparseAutoencoder
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS, face_boxes, read_csv
from render_face_region_heatmaps import load_region_boxes

DEFAULT_MODEL = ROOT / "sdxl-unbox/checkpoints/unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
PARTS = ("eyes", "eyebrows", "nose", "ears")
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")
TILE = 112
LABEL_H = 22
MAX_BOX_AREA = 0.12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    p.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    p.add_argument("--images", type=Path, default=ROOT / "backprop")
    p.add_argument("--scores", type=Path,
                   default=ROOT / "features-to-face/heldout_feature_validation.csv")
    p.add_argument("--output", type=Path,
                   default=ROOT / "features-to-face/seed_grids")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per-category", type=int, default=25)
    return p.parse_args()


def category(row):
    race = row["race_prompt"].replace(" ", "_")
    return f"{race}_{row['gender_prompt']}"


def choose_samples(rows, per_category):
    grouped = defaultdict(lambda: {"test": [], "validation": []})
    for row in rows:
        if row["split"] in ("test", "validation"):
            grouped[category(row)][row["split"]].append(row)
    chosen = {}
    for race in RACES:
        for gender in GENDERS:
            cat = f"{race}_{gender}"
            test = sorted(grouped[cat]["test"], key=lambda r: int(r["seed"]))
            val = sorted(grouped[cat]["validation"], key=lambda r: int(r["seed"]))
            if per_category != 25 or len(test) < 5 or len(val) < 20:
                raise ValueError(f"{cat}: need 20 validation and 5 test images for the 5x5 layout")
            # Spread the five test examples over the available test seed range.
            test_indices = np.rint(np.linspace(0, len(test)-1, 5)).astype(int)
            chosen[cat] = {"validation": val[:20],
                           "test": [test[index] for index in test_indices]}
    return chosen


def eligible_part_boxes(boxes, sample_id, part):
    eligible = []
    for item in boxes.get(sample_id, {}).get(part, []):
        x1, y1, x2, y2 = item["xyxy"]
        if 0 < (x2-x1)*(y2-y1)/(512*512) <= MAX_BOX_AREA:
            eligible.append(item)
    return eligible


def box_mask(boxes, height, width):
    xs = (np.arange(width, dtype=np.float32) + 0.5) * (512 / width)
    ys = (np.arange(height, dtype=np.float32) + 0.5) * (512 / height)
    mask = np.zeros((height, width), dtype=bool)
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def overlay(image, values, face_box_norm, target_boxes, part):
    height, width = values.shape
    face_box = tuple(float(coord) * 512 for coord in face_box_norm)
    face_mask = box_mask([{"xyxy": face_box}], height, width)
    visible_mask = np.ones_like(face_mask) if part == "ears" else face_mask
    visible = values[visible_mask]
    vmax = max(float(visible.max()) if visible.size else 0.0, 1e-8)
    normalized = np.clip(values / vmax, 0, 1) * visible_mask
    red = np.clip(1.5 - np.abs(4*normalized - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4*normalized - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4*normalized - 1), 0, 1)
    rgba = np.uint8(np.stack((red, green, blue, np.ones_like(normalized)), axis=-1) * 255)
    rgba[..., 3] = np.uint8(normalized * 175)
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


def load_score_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    args = parse_args()
    rows = read_csv(args.manifest)
    chosen = choose_samples(rows, args.per_category)
    region_boxes = load_region_boxes(args.boxes)
    faces = face_boxes(args.boxes)
    score_rows = load_score_rows(args.scores)
    features = sorted({int(row["feature"]) for row in score_rows})
    device = torch.device(args.device)
    model = SparseAutoencoder.load_from_disk(str(args.model_path)).float().to(device).eval()
    model.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for feature_index, feature in enumerate(features, 1):
        feature_dir = args.output / f"feature_{feature}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        pair_pages = {}
        for part in PARTS:
            rows_for_feature = [r for r in score_rows
                                if int(r["feature"]) == feature and r["region"] == part]
            score = rows_for_feature[0] if rows_for_feature else None
            pair_pages[part] = [f"# Feature {feature} — {part.title()}", "",
                                "Fixed original pretrained SAE; no fine-tuning. Each category grid is 5×5: the first four columns are 20 validation portraits and the last column is five test portraits. Cyan is the face box, yellow is the part box. A ‘No bbox detected’ tag means the automatic annotations supplied no usable box.", ""]
            if score:
                ci = f"[{float(score['test_bootstrap_ci_low']):.2f}, {float(score['test_bootstrap_ci_high']):.2f}]"
                pair_pages[part].extend([f"Held-out test contrast: {float(score['test_macro_contrast']):+.2f} (95% CI {ci}; n={score['test_images']}).", ""])
        for race in RACES:
            for gender in GENDERS:
                cat = f"{race}_{gender}"
                grids = {}
                for part in PARTS:
                    grid = Image.new("RGB", (5*TILE, 50 + 5*(TILE+LABEL_H)), "white")
                    draw = ImageDraw.Draw(grid)
                    draw.text((4, 4), f"{cat.replace('_', ' ').title()} · cols 1–4 validation · col 5 test",
                              fill="#26364a", font=font)
                    for col, text in enumerate(("VAL", "VAL", "VAL", "VAL", "TEST")):
                        draw.text((col*TILE+4, 25), text, fill="#26364a", font=font)
                    grids[part] = (grid, draw)
                selections = [(row, index % 4, index // 4)
                              for index, row in enumerate(chosen[cat]["validation"])]
                selections.extend((row, 4, index)
                                  for index, row in enumerate(chosen[cat]["test"]))
                for row, col, grid_row in selections:
                    sid = row["sample_id"]
                    residual_path = args.residuals / f"{sid}.npy"
                    image_path = args.images / row["image"]
                    x, y = col*TILE, 50 + grid_row*(TILE+LABEL_H)
                    if sid not in faces or not residual_path.is_file() or not image_path.is_file():
                        for _part, (_grid, draw) in grids.items():
                            draw.text((x+3, y+40), "MISSING IMAGE", fill="#a00000", font=font)
                        continue
                    residual = torch.from_numpy(np.load(residual_path, allow_pickle=False).astype(np.float32)).to(device)
                    with torch.no_grad():
                        values = model.encode(residual)[..., feature].float().cpu().numpy()
                    del residual
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    split_tag = "T" if row["split"] == "test" else "V"
                    for part, (grid, draw) in grids.items():
                        target_boxes = eligible_part_boxes(region_boxes, sid, part)
                        tile_image = overlay(image, values, faces[sid], target_boxes, part)
                        tile_image.thumbnail((TILE-2, TILE-2), Image.Resampling.LANCZOS)
                        grid.paste(tile_image, (x+1, y+1))
                        draw.text((x+2, y+TILE), f"{split_tag} seed {int(row['seed']):03d}",
                                  fill="#26364a", font=font)
                        if not target_boxes:
                            draw.rectangle((x+4, y+4, x+TILE-5, y+17), fill="#fff2a8")
                            draw.text((x+5, y+5), "NO BBOX DETECTED", fill="#222222", font=font)
                    del values
                for part, (grid, _draw) in grids.items():
                    part_dir = feature_dir / part
                    part_dir.mkdir(parents=True, exist_ok=True)
                    grid.save(part_dir / f"{cat}.jpg", quality=87, optimize=True)
                    pair_pages[part].extend([f"## {cat.replace('_', ' ').title()}", "",
                                             f"![Feature {feature} over {part} boxes for {cat}](feature_{feature}/{part}/{cat}.jpg)", ""])
        for part, content in pair_pages.items():
            (args.output / f"feature_{feature}_{part}.md").write_text("\n".join(content) + "\n")
        print(f"rendered feature {feature} ({feature_index}/{len(features)})", flush=True)

    pages = ["# Fixed-feature facial-part grids", "",
             "No training images are used. Each feature/part pair has a separate sheet page (for example, feature 527 has separate eyes and eyebrows pages). Each category grid is 5×5: first four columns show 20 validation images; last column shows five test images. The feature IDs were fixed from intervention runs before these splits were evaluated. Quantitative metrics continue to use all eligible test images, not just the five displayed.", ""]
    for feature in features:
        pages.extend([f"## Feature {feature}", "",
                      "| Part | Test contrast (95% CI) | Test n | Grid |",
                      "| --- | --- | ---: | --- |"])
        for part in PARTS:
            score = next((r for r in score_rows
                          if int(r["feature"]) == feature and r["region"] == part), None)
            if score:
                ci = f"[{float(score['test_bootstrap_ci_low']):.2f}, {float(score['test_bootstrap_ci_high']):.2f}]"
                value = f"{float(score['test_macro_contrast']):+.2f} {ci}"
                n = score["test_images"]
            else:
                value, n = "n/a", "n/a"
            pages.append(f"| {part.title()} | {value} | {n} | [view grid](feature_{feature}_{part}.md) |")
        pages.append("")
    (args.output / "index.md").write_text("\n".join(pages) + "\n")
    (args.output / "settings.json").write_text(json.dumps({
        "model": str(args.model_path), "feature_ids": features, "parts": list(PARTS),
        "images_per_category": args.per_category,
        "sample_mix": "20 validation images in columns 1-4; five test images sampled evenly across sorted test seed IDs in column 5; no training images",
        "test_only_quantitative_results": str(args.scores),
        "heatmap_normalization": "per image within face; full image for ears",
        "face_box_color": "cyan", "part_box_color": "yellow",
    }, indent=2) + "\n")
    print(f"saved feature grids to {args.output}", flush=True)


if __name__ == "__main__":
    main()
