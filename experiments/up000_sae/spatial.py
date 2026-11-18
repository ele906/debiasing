"""Compare each layer's independently selected features on matched discovery grids."""
import csv
import importlib.util
import json

import numpy as np
import torch

from discover import ROOT, RUN, SparseAutoencoder, group, write_csv

source = ROOT / "sae-training/fine_tune_face_sae.py"
spec = importlib.util.spec_from_file_location("face_training", source)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


@torch.inference_mode()
def main():
    mid = ROOT / "layers/runs/mid0_sae_face_001"
    mid_capture = ROOT / "layers/runs/mid0_sae_finetune_001/capture"
    data = ROOT / "sae-training/runs/face_parts_sae_001/test_extension"
    boxes = trainer.face_boxes(data / "annotations/bounding_boxes.csv")
    with (RUN / "capture/manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    analyses = [
        ("mid.0", mid / "sae", mid_capture,
         ROOT / "layers/mid0_stability/finetuned_ranking/top10.json"),
        ("up.0.0", RUN / "training/sae", RUN / "capture", RUN / "discovery/top10.json"),
    ]
    spatial = []
    for layer, checkpoint, capture, top in analyses:
        features = json.loads(top.read_text())["features"]
        sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
        for row in rows:
            sample = row["sample_id"]
            residual = torch.from_numpy(np.load(capture / "residuals" / f"{sample}.npy")).float().cuda()
            codes = sae.encode(residual)[..., features].cpu()
            mask = trainer.cell_mask(boxes[sample], 16, 16) if sample in boxes else None
            for index, feature in enumerate(features):
                grid = codes[..., index]
                total = grid.sum().item()
                spatial.append(dict(layer=layer, feature=feature, group=group(row), seed=int(row["seed"]),
                                    residual_rms=residual.square().mean().sqrt().item() if index == 0 else None,
                                    active_fraction=(grid > 0).float().mean().item(),
                                    mean=grid.mean().item(),
                                    face_area_fraction=mask.float().mean().item() if mask is not None else None,
                                    face_activation_fraction=(grid[mask].sum().item() / total
                                                              if mask is not None and total > 0 else None)))
    write_csv(RUN / "discovery/spatial_by_image.csv", spatial)


if __name__ == "__main__":
    main()
