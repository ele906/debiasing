"""Render Jet activation overlays for a face-SAE feature on new prompt-group images."""
import argparse
import json
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
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.uint8(np.stack((r, g, b), axis=-1) * 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "sae-training/runs/face_parts_sae_001/independent_test_seed300_309")
    parser.add_argument("--run", type=Path, default=ROOT / "sae-training/runs/face_sae_020")
    parser.add_argument("--feature", type=int, default=3725)
    parser.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/face_sae_020/group_analysis_001/feature_maps")
    args = parser.parse_args()
    data, run, output = args.data.resolve(), args.run.resolve(), args.output.resolve()
    manifest = list(__import__("csv").DictReader((data / "manifest.csv").open(newline="")))
    groups = ("black_woman", "white_woman", "east_asian_woman", "middle_eastern_woman", "indian_woman", "hispanic_woman")
    rows = [r for r in manifest if r["sample_id"].rsplit("_seed", 1)[0] in groups]
    if not rows:
        raise RuntimeError(f"No women-group images found in {data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SparseAutoencoder.load_from_disk(str(run / "sae")).float().to(device).eval()
    sae.requires_grad_(False)
    data_maps = {}
    shared_max = 0.0
    means = {}
    for row in rows:
        sid = row["sample_id"]
        residual = np.load(data / "residuals" / f"{sid}.npy", allow_pickle=False)
        with torch.no_grad():
            z = sae.encode(torch.from_numpy(residual[None]).to(device=device, dtype=torch.float32))[0, :, :, args.feature].cpu().numpy()
        data_maps[sid] = z
        means[sid] = float(z.mean())
        shared_max = max(shared_max, float(z.max()))

    seeds = sorted({int(r["seed"]) for r in rows})
    tile = 256
    sheet = Image.new("RGB", (len(groups) * tile, len(seeds) * (tile + 27) + 45), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), f"Face SAE feature {args.feature} | Jet overlays | shared scale across all panels", fill="black")
    for col, group in enumerate(groups):
        draw.text((col * tile + 4, 23), group.replace("_", " "), fill="black")
    for row_idx, seed in enumerate(seeds):
        for col, group in enumerate(groups):
            sid = f"{group}_seed{seed}"
            if sid not in data_maps:
                continue
            manifest_row = next(r for r in rows if r["sample_id"] == sid)
            image_path = data / manifest_row["image"]
            image = Image.open(image_path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            norm = np.clip(data_maps[sid] / max(shared_max, 1e-8), 0, 1)
            color = Image.fromarray(jet(norm), "RGB").resize((tile, tile), Image.Resampling.NEAREST)
            alpha = Image.fromarray(np.uint8(norm * 150), "L").resize((tile, tile), Image.Resampling.NEAREST)
            x, y = col * tile, 48 + row_idx * (tile + 27)
            sheet.paste(Image.composite(color, image, alpha), (x, y))
            draw.text((x + 4, y + tile + 2), f"seed {seed}", fill="black")

    output.mkdir(parents=True, exist_ok=True)
    image_out = output / f"feature_{args.feature}_women_seed300_309_jet.jpg"
    sheet.save(image_out, quality=93)
    summary = {
        "feature": args.feature,
        "images": len(rows),
        "seeds": [min(seeds), max(seeds)],
        "mean_activation_by_group": {
            group: float(np.mean([means[r["sample_id"]] for r in rows if r["sample_id"].rsplit("_seed", 1)[0] == group]))
            for group in groups
        },
        "overlay": "Jet, shared activation scale across all displayed images",
        "contact_sheet": str(image_out),
    }
    (output / f"feature_{args.feature}_women_seed300_309_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
