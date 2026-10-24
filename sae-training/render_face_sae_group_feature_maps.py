"""Render selected face-SAE features across matched prompt-group portraits."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder


def jet(values):
    x = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4*x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*x - 1), 0, 1)
    return np.uint8(np.stack((r, g, b), axis=-1) * 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "sae-training/runs/face_sae_020/group_analysis_001")
    parser.add_argument("--data", type=Path, default=ROOT / "sae-training/runs/face_parts_sae_001/independent_test_seed200_299")
    parser.add_argument("--run", type=Path, default=ROOT / "sae-training/runs/face_sae_020")
    parser.add_argument("--count", type=int, default=3, help="Top Indian-woman-associated features to render")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202, 203, 204])
    args = parser.parse_args()
    analysis, data, run = args.analysis.resolve(), args.data.resolve(), args.run.resolve()
    candidates = [r for r in csv.DictReader((analysis / "group_effects_top20.csv").open(newline=""))
                  if r["race_prompt"] == "indian" and r["gender_prompt"] == "woman"
                  and r["split"] == "discovery"]
    features = [int(r["feature"]) for r in candidates[:args.count]]
    with np.load(analysis / "face_sae_activations.npz", allow_pickle=False) as f:
        ids = f["sample_id"].astype(str)
    ids_set = set(ids)
    races = ("black", "white", "east asian", "middle eastern", "indian", "hispanic")
    groups = [race.replace(" ", "_") + "_woman" for race in races]
    images_dir, residual_dir = data / "images", data / "residuals"
    sae = SparseAutoencoder.load_from_disk(str(run / "sae")).float().cuda().eval()
    sae.requires_grad_(False)
    out = analysis / "feature_maps"
    out.mkdir(exist_ok=True)
    tile = 256
    for feature in features:
        maps, raw_images = {}, []
        max_value = 0.0
        for seed in args.seeds:
            for group in groups:
                sample_id = f"{group}_seed{seed}"
                if sample_id not in ids_set:
                    continue
                residual = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
                x = torch.from_numpy(residual[None]).float().cuda()
                with torch.no_grad():
                    z = sae.encode(x)[0, :, :, feature].float().cpu().numpy()
                max_value = max(max_value, float(z.max()))
                maps[sample_id] = z
                raw_images.append((sample_id, Image.open(images_dir / f"{group}/seed{seed}.png").convert("RGB")))
        sheet = Image.new("RGB", (len(groups) * tile, len(args.seeds) * (tile + 26) + 30), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), f"Face SAE feature {feature}; shared activation scale across groups/seeds", fill="black")
        for j, group in enumerate(groups):
            draw.text((j*tile + 4, 18), group.replace("_", " "), fill="black")
        for sample_id, image in raw_images:
            group, seed_text = sample_id.rsplit("_seed", 1)
            seed = int(seed_text)
            row = args.seeds.index(seed)
            col = groups.index(group)
            y = 44 + row * (tile + 26)
            image = image.resize((tile, tile), Image.Resampling.LANCZOS)
            norm = np.clip(maps[sample_id] / max(max_value, 1e-8), 0, 1)
            color = Image.fromarray(jet(norm), "RGB").resize((tile, tile), Image.Resampling.NEAREST)
            alpha = Image.fromarray(np.uint8(norm * 150), "L").resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(Image.composite(color, image, alpha), (col*tile, y))
            draw.text((col*tile + 4, y + tile + 2), f"seed {seed}", fill="black")
        sheet.save(out / f"feature_{feature}_women.jpg", quality=92)
    print(f"Wrote feature maps under {out}")


if __name__ == "__main__":
    main()
