"""Rank the fine-tuned up.0.0 dictionary over the historical discovery set."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder


def group(row):
    return row["race_prompt"].replace(" ", "_") + "_" + row["gender_prompt"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    with (RUN / "capture/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    groups = sorted({group(row) for row in rows})
    if len(groups) != 12 or any(
        sorted(int(row["seed"]) for row in rows if group(row) == g) != list(range(100, 200))
        for g in groups
    ):
        raise ValueError("Discovery requires all 100 seeds for each of 12 groups")

    output = RUN / "discovery"
    output.mkdir(exist_ok=True)
    sae = SparseAutoencoder.load_from_disk(str(RUN / "training/sae")).float().cuda().eval()
    sums = np.zeros((12, sae.n_dirs), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)

    def encode(row):
        path = RUN / "capture/residuals" / (row["sample_id"] + ".npy")
        return sae.encode(torch.from_numpy(np.load(path)).float().cuda()).cpu().numpy()

    for row in rows:
        index = groups.index(group(row))
        codes = encode(row)
        sums[index] += codes.mean(axis=(0, 1), dtype=np.float64)
        counts[index] += (codes > 0).sum(axis=(0, 1))
    means = sums / 100
    frequencies = counts / (100 * 256)
    candidates = {int(feature) for values in means for feature in np.argsort(values)[-10:]}
    ranking = []
    for feature, values in enumerate(means.T):
        peak = int(values.argmax())
        other = float(np.delete(values, peak).mean())
        active_count = counts[:, feature].sum()
        ranking.append(dict(feature=feature, peak_group=groups[peak], peak_mean=float(values[peak]),
                            other_mean=other, gap=float(values[peak] - other),
                            mean_activation=float(values.mean()),
                            activation_frequency=float(frequencies[:, feature].mean()),
                            mean_active_magnitude=float(sums[:, feature].sum() * 256 / max(active_count, 1)),
                            historical_candidate=feature in candidates))
    ranking.sort(key=lambda record: (-record["gap"], record["feature"]))
    selected = [record for record in ranking if record["historical_candidate"]][:10]
    write_csv(output / "full_ranking.csv", ranking)
    write_csv(output / "top10.csv", selected)
    np.savez_compressed(output / "group_activations.npz", groups=groups, means=means, frequencies=frequencies)
    overlays = output / "overlays"
    overlays.mkdir(exist_ok=True)
    source = ROOT / "sae-training/runs/face_parts_sae_001/test_extension"
    for record in selected:
        feature = record["feature"]
        maps = []
        for label in groups:
            row = next(row for row in rows if group(row) == label and row["seed"] == "185")
            maps.append((label, row, encode(row)[..., feature].copy()))
        scale = max(float(grid.max()) for _, _, grid in maps)
        for label, row, grid in maps:
            with Image.open(source / row["image"]) as original:
                image = original.convert("RGB")
            values = (grid / max(scale, 1e-12)).clip(0, 1)
            colors = np.stack([np.clip(1.5 - np.abs(4 * values - center), 0, 1)
                               for center in (3, 2, 1)], axis=-1)
            heat = Image.fromarray((colors * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
            alpha = Image.fromarray(((grid > 0) * 150).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
            Image.composite(heat, image, alpha).save(overlays / f"feature_{feature}_{label}_seed185.png")
    (output / "top10.json").write_text(json.dumps({"features": [record["feature"] for record in selected]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
