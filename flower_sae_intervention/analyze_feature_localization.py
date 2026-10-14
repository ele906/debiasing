"""Measure and visualize where the C2C-ranked SAE features activate."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_FEATURES = (940, 2810, 1855)


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def box_mask(boxes, h, w):
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        ix1 = max(0, min(w, int(np.floor(x1 * w))))
        ix2 = max(0, min(w, int(np.ceil(x2 * w))))
        iy1 = max(0, min(h, int(np.floor(y1 * h))))
        iy2 = max(0, min(h, int(np.ceil(y2 * h))))
        mask[iy1:iy2, ix1:ix2] = True
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=ROOT / "sae-training/c2c_box_targets.csv")
    p.add_argument("--manifest", type=Path, default=ROOT / "backprop/manifest.csv")
    p.add_argument("--residuals", type=Path, default=ROOT / "sae-training/residuals")
    p.add_argument("--sae-checkpoint", type=Path, default=ROOT / "sae-training/runs/c2c_box_sae_003/sae")
    p.add_argument("--images", type=Path, default=ROOT / "backprop")
    p.add_argument("--output", type=Path, default=HERE / "analysis")
    p.add_argument("--features", type=int, nargs="+", default=list(DEFAULT_FEATURES))
    p.add_argument("--limit", type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh directory: {args.output}")

    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ROOT / "backprop"))
    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from heatmap_style import make_heatmap_overlay
    from SAE import SparseAutoencoder

    device = torch.device("cuda")
    sae = SparseAutoencoder.load_from_disk(str(args.sae_checkpoint)).float().to(device).eval()
    targets = [r for r in rows(args.targets) if r["concept"] == "flower"
               and r["split"] == "test" and (r["race_prompt"], r["gender_prompt"]) == ("east asian", "woman")]
    manifests = {r["sample_id"]: r for r in rows(args.manifest)}
    selected = targets[:args.limit]
    args.output.mkdir(parents=True)
    records = []
    image_rows = []
    for row in selected:
        sid = row["sample_id"]
        residual = torch.from_numpy(np.load(args.residuals / f"{sid}.npy", allow_pickle=False).astype(np.float32)).to(device)
        h, w, d = residual.shape
        with torch.no_grad():
            z = sae.encode(residual.reshape(-1, d)).reshape(h, w, -1)
        z = z.cpu().numpy()
        boxes = json.loads(row["boxes_xyxy_normalized"])
        target = box_mask(boxes, h, w)
        image_path = args.images / manifests[sid]["image"]
        image = Image.open(image_path).convert("RGB")
        overlay = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle((x1*w, y1*h, x2*w, y2*h), outline=(0, 255, 0), width=1)
        for feature in args.features:
            act = z[..., feature]
            inside = float(act[target].mean()) if target.any() else float("nan")
            outside = float(act[~target].mean()) if (~target).any() else float("nan")
            peak = int(act.argmax())
            py, px = divmod(peak, w)
            records.append(dict(sample_id=sid, c2c_present=row["detector_present"], feature=feature,
                                mean_inside_box=inside, mean_outside_box=outside,
                                inside_outside_ratio=inside / max(outside, 1e-12) if target.any() else "",
                                peak_x_cell=px, peak_y_cell=py, peak_in_box=bool(target[py, px]) if target.any() else ""))
            image_rows.append((image_path, sid, feature, act.copy(), boxes))
        del residual, z

    with (args.output / "feature_localization.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    # Make maps for up to six positive held-out examples, with boxes and normalized activation heat.
    positive_ids = {r["sample_id"] for r in selected if int(r["detector_present"]) == 1}
    examples = []
    for path, sid, feature, act, boxes in image_rows:
        if sid not in positive_ids or len(examples) >= 18:
            continue
        base = Image.open(path).convert("RGB").resize((512, 512))
        composite = make_heatmap_overlay(base, act).convert("RGB")
        draw = ImageDraw.Draw(composite)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle((x1*512, y1*512, x2*512, y2*512), outline="white", width=3)
        draw.text((5, 5), f"{sid} | SAE feature {feature}", fill="white", stroke_width=2, stroke_fill="black")
        examples.append(composite)
    if examples:
        page = Image.new("RGB", (1024, ((len(examples)+1)//2)*512), "white")
        for i, im in enumerate(examples):
            page.paste(im, ((i % 2)*512, (i // 2)*512))
        page.save(args.output / "activation_overlays.jpg", quality=92)
    print(json.dumps(dict(output=str(args.output), samples=len(selected), positives=len(positive_ids),
                          features=args.features, grid_shape=[h, w]), indent=2))


if __name__ == "__main__":
    main()
