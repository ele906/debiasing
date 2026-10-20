"""Fine-tune a pretrained SAE using only activation cells inside eye boxes."""
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder
from SAE.sae import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fine_tune_face_sae import (
    BLOCK,
    DEFAULT_BOXES,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_RESIDUALS,
    SNAPSHOT_EPOCH,
    cell_mask,
    face_boxes,
    normalized_mse,
    read_csv,
    sha256,
    write_json,
)

RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")
MAX_BOX_AREA = 0.12
MIN_EYE_CELLS = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/eye_sae_001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot-interval", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Finish metrics/rankings from an existing completed checkpoint folder")
    args = parser.parse_args()
    if args.output.exists() and not args.evaluate_only:
        raise ValueError(f"Output exists; choose a fresh --output: {args.output}")
    if args.evaluate_only and not (args.output / "sae").is_dir():
        parser.error("--evaluate-only requires OUTPUT/sae from a completed training run")
    if args.epochs < 1 or args.lr <= 0 or args.snapshot_interval < 0:
        parser.error("epochs and learning rate must be positive; snapshot interval cannot be negative")
    return args


def read_eye_boxes(path):
    boxes = defaultdict(list)
    for row in read_csv(path):
        label = row["label"].strip().lower()
        if row["source"] not in {"objects", "grounding"}:
            continue
        if not re.search(r"\beyes?\b", label):
            continue
        coordinates = [float(row[key]) / 512.0 for key in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = coordinates
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area <= 0 or area > MAX_BOX_AREA:
            continue
        box = tuple(coordinates)
        if box not in boxes[row["sample_id"]]:
            boxes[row["sample_id"]].append(box)
    return dict(boxes)


def eye_cell_mask(sample_id, boxes, height, width):
    result = torch.zeros((height, width), dtype=torch.bool)
    for box in boxes.get(sample_id, []):
        result |= cell_mask(box, height, width)
    return result


def load_data(args):
    rows = read_csv(args.manifest)
    eyes = read_eye_boxes(args.boxes)
    faces = face_boxes(args.boxes)
    by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        sample_id = row["sample_id"]
        path = args.residuals / f"{sample_id}.npy"
        if row["split"] not in by_split or not path.is_file():
            continue
        if sample_id in eyes and sample_id in faces:
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
            eye_mask = eye_cell_mask(sample_id, eyes, shape[0], shape[1])
            if int(eye_mask.sum()) >= MIN_EYE_CELLS:
                by_split[row["split"]].append(row)
    if not by_split["train"] or not by_split["validation"] or not by_split["test"]:
        raise ValueError("Need eye-boxed train, validation, and test samples with enough SAE cells")
    return rows, eyes, faces, by_split


def make_balanced_epoch(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        category = (row["race_prompt"].replace(" ", "_"), row["gender_prompt"])
        groups[category].append(row["sample_id"])
    missing = [f"{race}_{gender}" for race in RACES for gender in GENDERS
               if (race, gender) not in groups]
    if missing:
        raise ValueError(f"Eye training data is missing categories: {', '.join(missing)}")
    per_category = math.ceil(len(rows) / len(groups))
    rng = random.Random(seed)
    balanced = [rng.choice(groups[key]) for key in sorted(groups)
                for _ in range(per_category)]
    rng.shuffle(balanced)
    return balanced


def load_tokens(sample_id, eyes, residual_dir, device):
    values = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
    residual = torch.from_numpy(values.astype(np.float32))
    mask = eye_cell_mask(sample_id, eyes, residual.shape[0], residual.shape[1])
    return residual.to(device), mask.to(device)


def train_epoch(sae, optimizer, sample_ids, eyes, residual_dir, device, epoch):
    sae.train()
    losses = []
    for step, sample_id in enumerate(sample_ids, 1):
        residual, mask = load_tokens(sample_id, eyes, residual_dir, device)
        eye_tokens = residual[mask]
        optimizer.zero_grad(set_to_none=True)
        reconstruction, info = sae(eye_tokens)
        scale = 1.0 / eye_tokens.var(unbiased=False).clamp_min(1e-8)
        main_loss = F.mse_loss(reconstruction, eye_tokens) * scale
        aux_reconstruction = sae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
        aux_target = eye_tokens - reconstruction.detach() + sae.pre_bias.detach()
        aux_loss = normalized_mse(aux_reconstruction, aux_target).nan_to_num(0.0)
        loss = main_loss + (1.0 / 32.0) * aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, sample {sample_id}")
        loss.backward()
        unit_norm_decoder_grad_adjustment_(sae)
        optimizer.step()
        with torch.no_grad():
            unit_norm_decoder_(sae)
        losses.append(float(loss.detach()))
        if step % 120 == 0:
            print(f"epoch={epoch} step={step}/{len(sample_ids)} loss={losses[-1]:.5f}", flush=True)
        del residual, mask, eye_tokens, reconstruction, info
        del aux_reconstruction, aux_target, main_loss, aux_loss, loss
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(sae, rows, eyes, faces, residual_dir, device):
    sae.eval()
    totals = defaultdict(lambda: [0.0, 0])
    categories = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for row in rows:
        sample_id = row["sample_id"]
        values = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
        residual = torch.from_numpy(values.astype(np.float32)).to(device)
        eye_mask = eye_cell_mask(sample_id, eyes, *residual.shape[:2]).to(device)
        face_mask = cell_mask(faces[sample_id], *residual.shape[:2]).to(device)
        reconstruction, _ = sae(residual.reshape(-1, residual.shape[-1]))
        error = (reconstruction.reshape_as(residual) - residual).square().sum(dim=-1)
        regions = {
            "eyes": eye_mask,
            "face_without_eyes": face_mask & ~eye_mask,
            "face": face_mask,
            "outside_face": ~face_mask,
            "all": torch.ones_like(face_mask),
        }
        category = f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"
        for region, mask in regions.items():
            sse = float(error[mask].sum())
            count = int(mask.sum()) * residual.shape[-1]
            totals[region][0] += sse
            totals[region][1] += count
            if count:
                categories[category][region][0] += sse
                categories[category][region][1] += count
        del residual, eye_mask, face_mask, reconstruction, error

    pooled = {region: sse / max(count, 1) for region, (sse, count) in totals.items()}
    per_category = {
        category: {region: sse / max(count, 1) for region, (sse, count) in values.items()}
        for category, values in categories.items()
    }
    macro = {
        region: float(np.mean([metrics[region] for metrics in per_category.values()
                               if region in metrics]))
        for region in totals
    }
    return {"images": len(rows), "pooled_mse": pooled,
            "macro_category_mse": macro, "per_category_mse": per_category}


@torch.no_grad()
def rank_eye_features(sae, rows, eyes, faces, residual_dir, device):
    sae.eval()
    eye_sum = torch.zeros(sae.n_dirs, device=device)
    other_face_sum = torch.zeros_like(eye_sum)
    eye_cells = other_face_cells = 0
    for row in rows:
        sample_id = row["sample_id"]
        residual = torch.from_numpy(
            np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False).astype(np.float32)
        ).to(device)
        eye_mask = eye_cell_mask(sample_id, eyes, *residual.shape[:2]).to(device)
        face_mask = cell_mask(faces[sample_id], *residual.shape[:2]).to(device)
        codes = sae.encode(residual)
        eye_sum += codes[eye_mask].sum(dim=0)
        other_face = face_mask & ~eye_mask
        other_face_sum += codes[other_face].sum(dim=0)
        eye_cells += int(eye_mask.sum())
        other_face_cells += int(other_face.sum())
    eye_mean = eye_sum / max(eye_cells, 1)
    other_mean = other_face_sum / max(other_face_cells, 1)
    contrast = eye_mean - other_mean
    order = torch.argsort(contrast, descending=True)[:20]
    sae.train()
    return [{
        "rank": rank + 1,
        "feature": int(index),
        "mean_eye_activation": float(eye_mean[index]),
        "mean_other_face_activation": float(other_mean[index]),
        "eye_minus_other_face": float(contrast[index]),
    } for rank, index in enumerate(order)]


def save_feature_ranking(path, ranking):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(ranking[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranking)


def save_initial_snapshot(sae, output_dir):
    snapshots = output_dir / "checkpoints"
    snapshots.mkdir()
    sae.save_to_disk(str(snapshots / SNAPSHOT_EPOCH.format(0)))
    return snapshots


def evaluate_checkpoint(base, tuned, rows, eyes, faces, residuals, device):
    return {
        "original_sae": evaluate(base, rows, eyes, faces, residuals, device),
        "eye_fine_tuned_sae": evaluate(tuned, rows, eyes, faces, residuals, device),
    }


def evaluate_saved_snapshots(run_dir, validation_rows, eyes, faces, residuals, device):
    snapshots_dir = run_dir / "checkpoints"
    snapshots = sorted(snapshots_dir.glob("epoch_[0-9][0-9][0-9]"))
    if not snapshots:
        raise ValueError(f"No checkpoints found in {snapshots_dir}")
    history = []
    best_epoch = None
    best_metric = float("inf")
    for snapshot in snapshots:
        model = SparseAutoencoder.load_from_disk(str(snapshot)).float().to(device).eval()
        metrics = evaluate(model, validation_rows, eyes, faces, residuals, device)
        epoch = int(snapshot.name.split("_")[1])
        history.append({"epoch": epoch, "train_loss_mean": None,
                        **metrics["pooled_mse"],
                        "macro_eye_mse": metrics["macro_category_mse"]["eyes"]})
        if epoch and metrics["macro_category_mse"]["eyes"] < best_metric:
            best_metric = metrics["macro_category_mse"]["eyes"]
            best_epoch = epoch
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return snapshots_dir, history, best_epoch


def write_run_outputs(args, rows, eyes, faces, split, base, selected,
                      snapshots_dir, best_epoch, history, device,
                      history_note=None):
    results = {}
    for name in ("validation", "test"):
        results[f"eye_labeled_{name}"] = evaluate_checkpoint(
            base, selected, split[name], eyes, faces, args.residuals, device
        )
        face_rows = [row for row in rows if row["split"] == name
                     and row["sample_id"] in faces
                     and (args.residuals / f"{row['sample_id']}.npy").is_file()]
        results[f"all_face_{name}"] = evaluate_checkpoint(
            base, selected, face_rows, eyes, faces, args.residuals, device
        )

    ranking = rank_eye_features(selected, split["validation"], eyes, faces,
                                args.residuals, device)
    save_feature_ranking(args.output / "eye_localized_features.csv", ranking)
    write_json(args.output / "training_history.json", history)
    write_json(args.output / "heldout_reconstruction.json", results)
    with (args.output / "heldout_metrics.csv").open("w", newline="") as target:
        fields = ("evaluation_set", "model", "images", "region", "pooled_mse",
                  "macro_category_mse")
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for evaluation_set, comparisons in results.items():
            for model_name, metrics in comparisons.items():
                for region, pooled_mse in metrics["pooled_mse"].items():
                    writer.writerow({
                        "evaluation_set": evaluation_set,
                        "model": model_name,
                        "images": metrics["images"],
                        "region": region,
                        "pooled_mse": pooled_mse,
                        "macro_category_mse": metrics["macro_category_mse"].get(region, ""),
                    })
    settings = {
        "experiment": "eye-region fine-tuning of pretrained Top-K SAE",
        "block": BLOCK,
        "checkpoint": str(args.checkpoint),
        "checkpoint_state_dict_sha256": sha256(args.checkpoint / "state_dict.pth"),
        "box_source": str(args.boxes),
        "box_source_type": "automatic object and phrase-grounding eye boxes; unreviewed",
        "box_filter": {"maximum_area_fraction": MAX_BOX_AREA,
                       "minimum_activation_cells": MIN_EYE_CELLS},
        "training_objective": "SAE reconstruction and AuxK loss on eye-box activation vectors only",
        "category_sampling": "uniform over race/gender categories, sampled with replacement to balance each epoch",
        "generator_updated": False,
        "sae_updated": True,
        "epochs": args.epochs,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_epochs": sorted(int(path.name.removeprefix("epoch_"))
                                   for path in snapshots_dir.iterdir()),
        "learning_rate": args.lr,
        "auxk_coefficient": 1.0 / 32.0,
        "seed": args.seed,
        "usable_eye_images_by_split": {name: len(items) for name, items in split.items()},
        "categories_by_split": {
            name: sorted({f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"
                          for row in items}) for name, items in split.items()
        },
        "best_validation_epoch": best_epoch,
        "selection_metric": "macro-average validation eye-region reconstruction MSE across race/gender categories",
        "interpretation": "Exploratory eye-region reconstruction specialization; boxes are sparse and unreviewed.",
    }
    if history_note:
        settings["training_history_note"] = history_note
    write_json(args.output / "settings.json", settings)
    print(json.dumps({"best_epoch": best_epoch, "heldout_reconstruction": results}, indent=2), flush=True)


def main():
    args = parse_args()
    rows, eyes, faces, split = load_data(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAE fine-tuning")
    device = torch.device("cuda")

    base = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).eval()
    if args.evaluate_only:
        snapshots_dir, history, best_epoch = evaluate_saved_snapshots(
            args.output, split["validation"], eyes, faces, args.residuals, device
        )
        selected = SparseAutoencoder.load_from_disk(str(args.output / "sae")).float().to(device).eval()
        write_run_outputs(
            args, rows, eyes, faces, split, base, selected, snapshots_dir,
            best_epoch, history, device,
            history_note="Validation metrics were recomputed at saved snapshot epochs; per-epoch training losses were printed during training but not persisted in this recovered run.",
        )
        return

    sae = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, eps=1e-8)
    args.output.mkdir(parents=True)
    snapshots_dir = save_initial_snapshot(sae, args.output)
    history = []
    best_validation = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        order = make_balanced_epoch(split["train"], args.seed + epoch)
        train_loss = train_epoch(sae, optimizer, order, eyes, args.residuals,
                                 device, epoch)
        val_metrics = evaluate(sae, split["validation"], eyes, faces,
                               args.residuals, device)
        row = {"epoch": epoch, "train_loss_mean": train_loss,
               **val_metrics["pooled_mse"],
               "macro_eye_mse": val_metrics["macro_category_mse"]["eyes"]}
        history.append(row)
        write_json(args.output / "training_history.json", history)
        print(json.dumps(row), flush=True)

        if row["macro_eye_mse"] < best_validation:
            best_validation = row["macro_eye_mse"]
            best_epoch = epoch
            sae.save_to_disk(str(args.output / "sae"))
        if args.snapshot_interval and epoch % args.snapshot_interval == 0:
            sae.save_to_disk(str(snapshots_dir / SNAPSHOT_EPOCH.format(epoch)))

    if args.snapshot_interval == 0 or args.epochs % args.snapshot_interval != 0:
        sae.save_to_disk(str(snapshots_dir / SNAPSHOT_EPOCH.format(args.epochs)))

    selected = SparseAutoencoder.load_from_disk(str(args.output / "sae")).float().to(device).eval()
    write_run_outputs(args, rows, eyes, faces, split, base, selected,
                      snapshots_dir, best_epoch, history, device)


if __name__ == "__main__":
    main()
