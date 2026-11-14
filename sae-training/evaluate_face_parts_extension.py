"""Evaluate the face-parts SAE on a separate, test-only image extension."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from fine_tune_face_sae import DEFAULT_CHECKPOINT, face_boxes, read_csv, write_json
from fine_tune_face_parts_sae import REGIONS, MIN_REGION_CELLS, evaluate, masks_for
from render_face_region_heatmaps import load_region_boxes
from SAE import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path(__file__).resolve().parent / "runs/face_parts_sae_001")
    parser.add_argument("--data", type=Path, required=True,
                        help="Supplemental dataset folder containing manifest.csv and residuals/")
    parser.add_argument("--boxes", type=Path,
                        help="Defaults to DATA/annotations/bounding_boxes.csv")
    parser.add_argument("--output", type=Path,
                        help="Defaults to DATA/face_parts_evaluation.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.boxes = (args.boxes or args.data / "annotations/bounding_boxes.csv").resolve()
    args.output = (args.output or args.data / "face_parts_evaluation.json").resolve()
    if not args.boxes.is_file():
        parser.error(f"Missing detector box CSV: {args.boxes}")

    rows = read_csv(args.data / "manifest.csv")
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("Expected a nonempty manifest where every sample is test-only")
    region_boxes = load_region_boxes(args.boxes)
    face_map = face_boxes(args.boxes)
    groups = {region: defaultdict(list) for region in REGIONS}
    coverage = {region: {"usable_images": 0, "categories": set()} for region in REGIONS}
    missing_residuals = 0
    face_box_images = 0
    for row in rows:
        sid = row["sample_id"]
        residual_path = args.data / "residuals" / f"{sid}.npy"
        if not residual_path.is_file():
            missing_residuals += 1
            continue
        if sid not in face_map:
            continue
        face_box_images += 1
        shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
        masks = masks_for(sid, region_boxes, shape, face_map)
        category = row["race_prompt"].replace(" ", "_") + "_" + row["gender_prompt"]
        for region in REGIONS:
            if int(masks[region].sum()) < MIN_REGION_CELLS:
                continue
            groups[region][category].append(row)
            coverage[region]["usable_images"] += 1
            coverage[region]["categories"].add(category)

    if any(not groups[region] for region in REGIONS):
        raise ValueError("At least one region has no usable boxes; inspect detector output")
    device = torch.device(args.device)
    base = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).eval()
    tuned = SparseAutoencoder.load_from_disk(str(args.run / "sae")).float().to(device).eval()
    results = {
        "dataset": str(args.data),
        "images": len(rows),
        "images_with_face_box": face_box_images,
        "missing_residual_files": missing_residuals,
        "annotation_source": str(args.boxes),
        "annotations_reviewed_by_human": False,
        "regions": {},
        "original_sae": evaluate(base, groups, region_boxes, face_map,
                                 args.data / "residuals", device),
        "multi_region_fine_tuned_sae": evaluate(tuned, groups, region_boxes, face_map,
                                                  args.data / "residuals", device),
    }
    for region in REGIONS:
        results["regions"][region] = {
            "usable_images": coverage[region]["usable_images"],
            "usable_categories": sorted(coverage[region]["categories"]),
        }
    write_json(args.output, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
