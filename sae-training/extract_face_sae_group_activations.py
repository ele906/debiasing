"""Encode held-out face residuals with the saved face-region SAE."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_face_boxes(path):
    boxes = {}
    for row in read_csv(path):
        if row["label"].strip().lower() != "human face":
            continue
        xyxy = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            boxes.setdefault(row["sample_id"], []).append(xyxy)
    return boxes


def mask_for_boxes(boxes, height=16, width=16):
    xs = (np.arange(width) + 0.5) * (512 / width)
    ys = (np.arange(height) + 0.5) * (512 / height)
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "sae-training/runs/face_parts_sae_001/independent_test_seed200_299")
    parser.add_argument("--run", type=Path, default=ROOT / "sae-training/runs/face_sae_020")
    parser.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/face_sae_020/group_analysis_001")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    data, run, out = args.data.resolve(), args.run.resolve(), args.output.resolve()
    residual_dir = data / "residuals"
    manifest = read_csv(data / "manifest.csv")
    boxes = load_face_boxes(data / "annotations/bounding_boxes.csv")
    rows = [r for r in manifest if (residual_dir / f"{r['sample_id']}.npy").is_file()]
    if not rows:
        raise RuntimeError(f"No residuals found under {residual_dir}")
    out.mkdir(parents=True, exist_ok=True)
    model_dir = run / "sae"
    weight_path = model_dir / "state_dict.pth"
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SparseAutoencoder.load_from_disk(str(model_dir)).float().to(device).eval()
    sae.requires_grad_(False)
    n, d = len(rows), sae.n_dirs
    face_mean = np.full((n, d), np.nan, dtype=np.float32)
    face_max = np.full((n, d), np.nan, dtype=np.float32)
    face_active = np.full((n, d), np.nan, dtype=np.float32)
    global_mean = np.zeros((n, d), dtype=np.float32)
    usable = np.zeros(n, dtype=bool)
    masks = [mask_for_boxes(boxes.get(r["sample_id"], [])) for r in rows]

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        residuals = np.stack([np.load(residual_dir / f"{rows[i]['sample_id']}.npy", allow_pickle=False)
                              for i in range(start, end)])
        x = torch.from_numpy(residuals).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            z = sae.encode(x).float().cpu().numpy()
        for offset, i in enumerate(range(start, end)):
            global_mean[i] = z[offset].mean(axis=(0, 1))
            mask = masks[i]
            if mask.any():
                values = z[offset][mask]
                face_mean[i] = values.mean(axis=0)
                face_max[i] = values.max(axis=0)
                face_active[i] = (values > 0).mean(axis=0)
                usable[i] = True
        if end == n or end % 128 == 0:
            print(f"encoded {end}/{n} residuals; face boxes usable {int(usable[:end].sum())}", flush=True)
        del x, z

    np.savez_compressed(out / "face_sae_activations.npz",
                        X_face_mean=face_mean, X_face_max=face_max,
                        X_face_active_fraction=face_active, X_global_mean=global_mean,
                        sample_id=np.array([r["sample_id"] for r in rows]),
                        race=np.array([r["race_prompt"] for r in rows]),
                        gender=np.array([r["gender_prompt"] for r in rows]),
                        seed=np.array([int(r["seed"]) for r in rows]),
                        split=np.array([r["split"] for r in rows]),
                        face_box_usable=usable)
    settings = {
        "experiment": "group activation scan on existing held-out portraits",
        "sae_run": str(run), "sae_weights_sha256": weight_hash,
        "activation_block": "unet.down_blocks.2.attentions.1",
        "residual_source": str(residual_dir), "residual_source_sae": "pretrained SDXL Turbo SAE",
        "encoding_model": "face_sae_020 selected epoch 20",
        "pooling": "mean, max, and active fraction within automatic human-face box; plus global mean",
        "box_source": str(data / "annotations/bounding_boxes.csv"),
        "images": n, "face_box_usable": int(usable.sum()),
        "labels": "race_prompt and gender_prompt are generated prompt groups",
        "device": str(device), "batch_size": args.batch_size,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings), flush=True)


if __name__ == "__main__":
    main()
