"""Fine-tune one pretrained SAE on balanced eyes, eyebrows, nose, and ears."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from SAE.sae import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
from fine_tune_face_sae import (
    BLOCK, DEFAULT_BOXES, DEFAULT_CHECKPOINT, DEFAULT_MANIFEST, DEFAULT_RESIDUALS,
    cell_mask, face_boxes, normalized_mse, read_csv, sha256, write_json,
)
from render_face_region_heatmaps import load_region_boxes

REGIONS = ("eyes", "eyebrows", "nose", "ears")
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")
MAX_BOX_AREA = 0.12
MIN_REGION_CELLS = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "sae-training/runs/face_parts_sae_001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot-interval", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists; choose a fresh --output: {args.output}")
    if args.epochs < 1 or args.lr <= 0 or args.snapshot_interval < 0:
        parser.error("epochs and learning rate must be positive; snapshot interval cannot be negative")
    return args


def category(row):
    return row["race_prompt"].replace(" ", "_") + "_" + row["gender_prompt"]


def masks_for(sample_id, region_boxes, shape, face_map):
    height, width = shape[:2]
    masks = {}
    face = cell_mask(face_map[sample_id], height, width)
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) * (512 / width)
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) * (512 / height)
    for region in REGIONS:
        mask = torch.zeros((height, width), dtype=torch.bool)
        for item in region_boxes.get(sample_id, {}).get(region, []):
            x1, y1, x2, y2 = item["xyxy"]
            area = (x2 - x1) * (y2 - y1) / (512 * 512)
            if area > MAX_BOX_AREA:
                continue
            mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                     & (ys[:, None] >= y1) & (ys[:, None] <= y2))
        # Facial parts are restricted to the face box; ears may sit beyond it.
        masks[region] = mask if region == "ears" else mask & face
    return masks


def load_splits(args):
    rows = read_csv(args.manifest)
    region_boxes = load_region_boxes(args.boxes)
    face_map = face_boxes(args.boxes)
    splits = {name: {region: defaultdict(list) for region in REGIONS}
              for name in ("train", "validation", "test")}
    mask_cache = {}
    for row in rows:
        split, sid = row["split"], row["sample_id"]
        residual_path = args.residuals / f"{sid}.npy"
        if split not in splits or sid not in face_map or not residual_path.is_file():
            continue
        shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
        masks = masks_for(sid, region_boxes, shape, face_map)
        for region in REGIONS:
            if int(masks[region].sum()) >= MIN_REGION_CELLS:
                splits[split][region][category(row)].append(row)
    for split in splits:
        absent = [region for region in REGIONS if not splits[split][region]]
        if absent:
            raise ValueError(f"No usable {split} examples for: {', '.join(absent)}")
    return rows, region_boxes, face_map, splits


def balanced_epoch(train_groups, seed):
    """Use equal region totals, with equal sampling over available categories."""
    rng = random.Random(seed)
    target = max(sum(len(items) for items in groups.values()) for groups in train_groups.values())
    samples = []
    for region in REGIONS:
        groups = train_groups[region]
        keys = sorted(groups)
        per_category = math.ceil(target / len(keys))
        selected = []
        for key in keys:
            selected.extend((region, rng.choice(groups[key])) for _ in range(per_category))
        rng.shuffle(selected)
        samples.extend(selected[:target])
    rng.shuffle(samples)
    return samples


def load_tokens(row, region, region_boxes, face_map, residual_dir, device):
    sid = row["sample_id"]
    residual = torch.from_numpy(
        np.load(residual_dir / f"{sid}.npy", allow_pickle=False).astype(np.float32)
    )
    mask = masks_for(sid, region_boxes, residual.shape, face_map)[region]
    return residual.to(device), mask.to(device)


def train_epoch(sae, optimizer, samples, region_boxes, face_map, residual_dir, device, epoch):
    sae.train()
    losses = []
    region_counts = defaultdict(int)
    for step, (region, row) in enumerate(samples, 1):
        residual, mask = load_tokens(row, region, region_boxes, face_map,
                                     residual_dir, device)
        tokens = residual[mask]
        optimizer.zero_grad(set_to_none=True)
        reconstruction, info = sae(tokens)
        scale = 1.0 / tokens.var(unbiased=False).clamp_min(1e-8)
        main_loss = F.mse_loss(reconstruction, tokens) * scale
        aux_reconstruction = sae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
        aux_target = tokens - reconstruction.detach() + sae.pre_bias.detach()
        aux_loss = normalized_mse(aux_reconstruction, aux_target).nan_to_num(0.0)
        loss = main_loss + (1.0 / 32.0) * aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, sample {row['sample_id']}")
        loss.backward()
        unit_norm_decoder_grad_adjustment_(sae)
        optimizer.step()
        with torch.no_grad():
            unit_norm_decoder_(sae)
        losses.append(float(loss.detach()))
        region_counts[region] += 1
        if step % 120 == 0:
            print(f"epoch={epoch} step={step}/{len(samples)} loss={losses[-1]:.5f}", flush=True)
        del residual, mask, tokens, reconstruction, info, aux_reconstruction, aux_target, loss
    return float(np.mean(losses)), dict(region_counts)


@torch.no_grad()
def evaluate(model, groups, region_boxes, face_map, residual_dir, device):
    model.eval()
    per_region_category = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for region in REGIONS:
        seen = set()
        for rows in groups[region].values():
            for row in rows:
                sid = row["sample_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                residual, mask = load_tokens(row, region, region_boxes, face_map,
                                             residual_dir, device)
                recon, _ = model(residual.reshape(-1, residual.shape[-1]))
                error = (recon.reshape_as(residual) - residual).square().sum(dim=-1)
                n = int(mask.sum()) * residual.shape[-1]
                bucket = per_region_category[region][category(row)]
                bucket[0] += float(error[mask].sum())
                bucket[1] += n
                del residual, mask, recon, error
    result = {}
    for region in REGIONS:
        by_category = per_region_category[region]
        category_mse = {key: sse / max(n, 1) for key, (sse, n) in by_category.items()}
        pooled_sse = sum(sse for sse, _ in by_category.values())
        pooled_n = sum(n for _, n in by_category.values())
        result[region] = {
            "images": sum(len(set(row["sample_id"] for row in rows))
                          for rows in groups[region].values()),
            "pooled_mse": pooled_sse / max(pooled_n, 1),
            "macro_category_mse": float(np.mean(list(category_mse.values()))),
            "categories": len(category_mse),
            "per_category_mse": category_mse,
        }
    result["macro_region_mse"] = float(np.mean([
        result[region]["macro_category_mse"] for region in REGIONS
    ]))
    return result


@torch.no_grad()
def rank_features(model, groups, region_boxes, face_map, residual_dir, device):
    model.eval()
    output = {}
    for region in REGIONS:
        part_sum = torch.zeros(model.n_dirs, device=device)
        rest_sum = torch.zeros_like(part_sum)
        part_cells = rest_cells = 0
        seen = set()
        for rows in groups[region].values():
            for row in rows:
                sid = row["sample_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                residual, mask = load_tokens(row, region, region_boxes, face_map,
                                             residual_dir, device)
                codes = model.encode(residual)
                if region == "ears":
                    reference = torch.ones_like(mask)
                else:
                    reference = cell_mask(face_map[sid], *residual.shape[:2]).to(device)
                rest = reference & ~mask
                part_sum += codes[mask].sum(dim=0)
                rest_sum += codes[rest].sum(dim=0)
                part_cells += int(mask.sum())
                rest_cells += int(rest.sum())
        part_mean, rest_mean = part_sum / max(part_cells, 1), rest_sum / max(rest_cells, 1)
        contrast = part_mean - rest_mean
        best = torch.argsort(contrast, descending=True)[:20]
        output[region] = [{
            "region": region, "rank": rank + 1, "feature": int(index),
            "mean_region_activation": float(part_mean[index]),
            "mean_rest_activation": float(rest_mean[index]),
            "region_minus_rest": float(contrast[index]),
        } for rank, index in enumerate(best)]
    return output


def main():
    args = parse_args()
    rows, region_boxes, face_map, splits = load_splits(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAE fine-tuning")
    device = torch.device("cuda")
    base = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).eval()
    sae = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, eps=1e-8)
    args.output.mkdir(parents=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    sae.save_to_disk(str(checkpoints / "epoch_000"))
    history = []
    best_score, best_epoch = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        samples = balanced_epoch(splits["train"], args.seed + epoch)
        train_loss, region_steps = train_epoch(
            sae, optimizer, samples, region_boxes, face_map, args.residuals, device, epoch
        )
        validation = evaluate(sae, splits["validation"], region_boxes, face_map,
                              args.residuals, device)
        entry = {"epoch": epoch, "train_loss_mean": train_loss,
                 "train_steps_by_region": region_steps,
                 "validation": validation,
                 "selection_metric_macro_region_mse": validation["macro_region_mse"]}
        history.append(entry)
        write_json(args.output / "training_history.json", history)
        print(json.dumps(entry), flush=True)
        if validation["macro_region_mse"] < best_score:
            best_score, best_epoch = validation["macro_region_mse"], epoch
            sae.save_to_disk(str(args.output / "sae"))
        if args.snapshot_interval and epoch % args.snapshot_interval == 0:
            sae.save_to_disk(str(checkpoints / f"epoch_{epoch:03d}"))

    if args.snapshot_interval == 0 or args.epochs % args.snapshot_interval:
        sae.save_to_disk(str(checkpoints / f"epoch_{args.epochs:03d}"))
    selected = SparseAutoencoder.load_from_disk(str(args.output / "sae")).float().to(device).eval()
    heldout = {}
    for split in ("validation", "test"):
        heldout[split] = {
            "original_sae": evaluate(base, splits[split], region_boxes, face_map,
                                     args.residuals, device),
            "multi_region_fine_tuned_sae": evaluate(
                selected, splits[split], region_boxes, face_map, args.residuals, device
            ),
        }
    write_json(args.output / "heldout_reconstruction.json", heldout)
    rankings = rank_features(selected, splits["validation"], region_boxes, face_map,
                             args.residuals, device)
    with (args.output / "region_localized_features.csv").open("w", newline="") as target:
        fields = ("region", "rank", "feature", "mean_region_activation",
                  "mean_rest_activation", "region_minus_rest")
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for region in REGIONS:
            writer.writerows(rankings[region])
    split_counts = {
        split: {region: {key: len(values) for key, values in groups[region].items()}
                for region in REGIONS}
        for split, groups in splits.items()
    }
    settings = {
        "experiment": "single shared SAE fine-tuned on four facial regions",
        "block": BLOCK,
        "architecture": "pretrained Top-K SAE; shared encoder/decoder; no CNN or U-Net update",
        "checkpoint": str(args.checkpoint),
        "checkpoint_state_dict_sha256": sha256(args.checkpoint / "state_dict.pth"),
        "box_source": str(args.boxes),
        "box_source_type": "automatic object and phrase-grounding boxes; unreviewed",
        "regions": list(REGIONS),
        "box_filter": {"maximum_area_fraction": MAX_BOX_AREA,
                       "minimum_activation_cells": MIN_REGION_CELLS},
        "training_objective": "SAE reconstruction and AuxK loss on selected region cells",
        "sampler": "equal number of training steps per region; sample categories uniformly within each region",
        "generator_updated": False,
        "sae_updated": True,
        "epochs": args.epochs,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_epochs": sorted(int(path.name.removeprefix("epoch_"))
                                   for path in checkpoints.iterdir()),
        "learning_rate": args.lr,
        "auxk_coefficient": 1 / 32,
        "seed": args.seed,
        "usable_images_by_split_region_category": split_counts,
        "best_validation_epoch": best_epoch,
        "selection_metric": "mean across regions of macro race/gender category MSE on each region's held-out cells",
        "interpretation": "Exploratory reconstruction specialization; detections are automatic and unreviewed.",
    }
    write_json(args.output / "settings.json", settings)
    print(json.dumps({"best_epoch": best_epoch, "heldout_reconstruction": heldout}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
