"""Render selected features at three fixed discovery seeds per prompt group."""
import csv
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from discover import ROOT, RUN, SparseAutoencoder, group

SEEDS = (185, 192, 199)
TILE = 132


def overlay(image, grid, scale):
    values = (grid / max(scale, 1e-12)).clip(0, 1)
    colors = np.stack([np.clip(1.5 - np.abs(4 * values - center), 0, 1)
                       for center in (3, 2, 1)], axis=-1)
    heat = Image.fromarray((255 * colors).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    alpha = Image.fromarray(((grid > 0) * 150).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    return Image.composite(heat, image, alpha)


@torch.inference_mode()
def main():
    output = RUN / "multiseed_jet"
    output.mkdir(exist_ok=True)
    source = ROOT / "sae-training/runs/face_parts_sae_001/test_extension"
    with (RUN / "capture/manifest.csv").open(newline="") as stream:
        rows = sorted((row for row in csv.DictReader(stream) if int(row["seed"]) in SEEDS),
                      key=lambda row: (group(row), int(row["seed"])))
    if len(rows) != 36:
        raise ValueError("Expected three fixed seeds across each of the 12 prompt groups")
    analyses = {
        "up.0.0": (RUN / "training/sae", RUN / "capture", RUN / "discovery/top10.json"),
        "mid.0": (ROOT / "layers/runs/mid0_sae_face_001/sae",
                  ROOT / "layers/runs/mid0_sae_finetune_001/capture",
                  ROOT / "layers/mid0_stability/finetuned_ranking/top10.json"),
    }
    scales = {}
    for layer, (checkpoint, capture, ranking) in analyses.items():
        features = json.loads(ranking.read_text())["features"]
        sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
        grids = []
        faces = []
        for row in rows:
            residual = np.load(capture / "residuals" / (row["sample_id"] + ".npy"))
            codes = sae.encode(torch.from_numpy(residual).float().cuda())
            grids.append(codes[..., features].cpu().numpy())
            with Image.open(source / row["image"]) as image:
                faces.append(image.convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS))
        grids = np.asarray(grids)
        scales[layer] = {}
        for rank, feature in enumerate(features):
            scale = float(grids[..., rank].max())
            scales[layer][feature] = scale
            panel_width = 118 + 3 * TILE
            sheet = Image.new("RGB", (2 * panel_width + 10, 62 + 6 * TILE), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f"{layer} | Feature {feature} | Disparity rank {rank + 1}", fill="#142539")
            draw.text((8, 24), f"Shared JET scale 0 to {scale:.1f} within this feature. Inactive cells are transparent.",
                      fill="#364c60")
            for index, (row, image) in enumerate(zip(rows, faces)):
                group_index, seed_index = divmod(index, 3)
                panel_index, slot = divmod(group_index, 6)
                left = panel_index * panel_width
                x = left + 118 + seed_index * TILE
                y = 62 + slot * TILE
                if slot == 0:
                    draw.text((x + 5, 46), f"Seed {row['seed']}", fill="#142539")
                if seed_index == 0:
                    draw.text((left + 6, y + TILE // 2), group(row).replace("_", " "), fill="#142539")
                grid = grids[index, ..., rank]
                sheet.paste(overlay(image, grid, scale), (x, y))
            sheet.save(output / f"{layer}_feature_{feature}.jpg", quality=88)
    (output / "settings.json").write_text(json.dumps({"seeds": SEEDS, "feature_maxima": scales}, indent=2) + "\n")


if __name__ == "__main__":
    main()
