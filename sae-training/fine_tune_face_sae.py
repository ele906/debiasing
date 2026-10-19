"""Fine-tune a copy of the pretrained SAE on activation cells inside face boxes.

This does not train C2C or the image generator. C2C object-detection boxes are
used only as spatial masks and remain noisy, provisional labels.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdxl-unbox"))
from SAE import SparseAutoencoder
from SAE.sae import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESIDUALS = ROOT / "sae-training/residuals"
DEFAULT_MANIFEST = ROOT / "backprop/manifest.csv"
DEFAULT_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
DEFAULT_CHECKPOINT = ROOT / "sdxl-unbox/checkpoints/unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
BLOCK = "unet.down_blocks.2.attentions.1"
SPLITS = ("train", "validation", "test")
SNAPSHOT_EPOCH = "epoch_{:03d}"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def face_boxes(path):
    result = {}
    for row in read_csv(path):
        is_face_detection = (
            row["source"] == "objects"
            and row["label"].strip().lower() == "human face"
        )
        if not is_face_detection:
            continue
        box = [float(row[key]) / 512.0 for key in ("x1", "y1", "x2", "y2")]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result[row["sample_id"]] = box
    return result


def cell_mask(box, height, width):
    x1, y1, x2, y2 = box
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    return (
        (ys[:, None] >= y1)
        & (ys[:, None] <= y2)
        & (xs[None, :] >= x1)
        & (xs[None, :] <= x2)
    )


def load_face_tokens(sample_id, boxes, residual_dir, device):
    residual_path = residual_dir / f"{sample_id}.npy"
    array = np.load(residual_path, allow_pickle=False).astype(np.float32)
    residual = torch.from_numpy(array)
    h, w, d = residual.shape
    mask = cell_mask(boxes[sample_id], h, w)
    return residual.to(device), mask.to(device)


def reconstruction(sae, flat):
    rec, _ = sae(flat)
    return rec


def normalized_mse(recon, target):
    mean = target.mean(dim=0, keepdim=True)
    baseline = F.mse_loss(mean.expand_as(target), target).clamp_min(1e-8)
    return F.mse_loss(recon, target) / baseline


def train_epoch(
    sae,
    optimizer,
    sample_ids,
    boxes,
    residual_dir,
    device,
    epoch,
    normalize_decoder,
    adjust_decoder_gradients,
):
    sae.train()
    losses = []

    for step, sample_id in enumerate(sample_ids, start=1):
        residual, mask = load_face_tokens(sample_id, boxes, residual_dir, device)
        face_tokens = residual[mask]

        optimizer.zero_grad(set_to_none=True)
        reconstruction, info = sae(face_tokens)
        variance_scale = 1.0 / face_tokens.var(unbiased=False).clamp_min(1e-8)
        main_loss = F.mse_loss(reconstruction, face_tokens) * variance_scale
        aux_reconstruction = sae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
        aux_target = face_tokens - reconstruction.detach() + sae.pre_bias.detach()
        aux_loss = normalized_mse(aux_reconstruction, aux_target).nan_to_num(0.0)
        loss = main_loss + (1.0 / 32.0) * aux_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, image {sample_id}")

        loss.backward()
        adjust_decoder_gradients(sae)
        optimizer.step()
        with torch.no_grad():
            normalize_decoder(sae)

        losses.append(float(loss.detach()))
        if step % 120 == 0:
            print(
                f"epoch={epoch} step={step}/{len(sample_ids)} loss={losses[-1]:.5f}",
                flush=True,
            )
        del residual, mask, face_tokens, reconstruction, info
        del aux_reconstruction, aux_target, main_loss, aux_loss, loss

    return float(np.mean(losses))


def save_snapshot(sae, snapshot_dir, epoch):
    sae.save_to_disk(str(snapshot_dir / SNAPSHOT_EPOCH.format(epoch)))


def load_training_data(args):
    rows = read_csv(args.manifest)
    boxes = face_boxes(args.boxes)
    eligible = [
        row for row in rows
        if row["sample_id"] in boxes
        and (args.residuals / f"{row['sample_id']}.npy").is_file()
    ]
    splits = {
        name: [row["sample_id"] for row in eligible if row["split"] == name]
        for name in SPLITS
    }
    if not all(splits.values()):
        raise ValueError("Need face-boxed train, validation, and test samples")
    return boxes, eligible, splits


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_splits(base, selected, splits, boxes, residual_dir, device):
    results = {}
    for name in ("validation", "test"):
        base_metrics, base_counts = evaluate(
            base, splits[name], boxes, residual_dir, device
        )
        tuned_metrics, tuned_counts = evaluate(
            selected, splits[name], boxes, residual_dir, device
        )
        results[name] = {
            "images": len(splits[name]),
            "original_sae": base_metrics,
            "face_fine_tuned_sae": tuned_metrics,
            "token_counts": {
                "original": base_counts,
                "fine_tuned": tuned_counts,
            },
        }
    return results


def write_feature_ranking(path, feature_ranking):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(feature_ranking[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(feature_ranking)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "sae-training/runs/face_sae_001"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=4,
        help="Save SAE checkpoints every N epochs; use 0 to disable periodic snapshots",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh --output: {args.output}")
    if args.epochs < 1 or args.lr <= 0 or args.snapshot_interval < 0:
        parser.error("epochs and learning rate must be positive; snapshot interval cannot be negative")
    return args


def build_run_settings(args, checkpoint_hash, split, best_epoch, snapshots_dir):
    return {
        "experiment": "face-region fine-tuning of pretrained Top-K SAE",
        "block": BLOCK,
        "checkpoint": str(args.checkpoint),
        "checkpoint_state_dict_sha256": checkpoint_hash,
        "box_source": str(args.boxes),
        "box_source_type": "C2C/Florence-2 object detection; label human face; unreviewed",
        "phrase_grounding_used": False,
        "mask": "SAE input vectors at 16x16 cell centers inside detector face box",
        "generator_updated": False,
        "sae_updated": True,
        "epochs": args.epochs,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_epochs": sorted(
            int(path.name.removeprefix("epoch_")) for path in snapshots_dir.iterdir()
        ),
        "learning_rate": args.lr,
        "auxk_coefficient": 1.0 / 32.0,
        "seed": args.seed,
        "eligible_images_by_split": {
            name: len(sample_ids) for name, sample_ids in split.items()
        },
        "best_validation_epoch": best_epoch,
        "selection_metric": "validation face-region reconstruction MSE",
        "interpretation": "Exploratory face-region specialization; detector boxes are broad and unreviewed.",
    }


def make_feature_overlay(image, activation_map):
    values = activation_map.float().cpu().numpy()
    scale = max(float(values.max()), 1e-6)
    alpha = Image.fromarray(
        np.uint8(np.clip(values / scale, 0, 1) * 255)
    ).resize(image.size)
    red = Image.new("RGB", image.size, "red")
    return Image.composite(red, image, alpha.point(lambda value: int(value * 0.55)))


def make_heatmap_sheet(group, images, tile_size):
    tile_width, tile_height = tile_size
    sheet = Image.new("RGB", (tile_width * len(images), tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, image) in enumerate(images):
        x = column * tile_width
        sheet.paste(image.resize((tile_width, tile_width)), (x, 20))
        draw.text((x + 3, 3), f"{group}: {label}", fill="black")
    return sheet


@torch.no_grad()
def evaluate(sae, ids, boxes, residual_dir, device):
    sae.eval()
    sums = {
        "face_sse": 0.0,
        "face_n": 0,
        "outside_sse": 0.0,
        "outside_n": 0,
        "all_sse": 0.0,
        "all_n": 0,
    }
    for sid in ids:
        residual, mask = load_face_tokens(sid, boxes, residual_dir, device)
        flat = residual.reshape(-1, residual.shape[-1])
        rec = reconstruction(sae, flat).reshape_as(residual)
        err = (rec - residual).square().sum(dim=-1)
        outside = ~mask
        regions = (
            ("face", mask),
            ("outside", outside),
            ("all", torch.ones_like(mask)),
        )
        for name, selected in regions:
            sums[f"{name}_sse"] += float(err[selected].sum())
            sums[f"{name}_n"] += int(selected.sum()) * residual.shape[-1]
        del residual, mask, flat, rec, err
    sae.train()
    metrics = {
        f"{area}_mse": sums[f"{area}_sse"] / max(sums[f"{area}_n"], 1)
        for area in ("face", "outside", "all")
    }
    return metrics, sums


@torch.no_grad()
def rank_face_features(sae, ids, boxes, residual_dir, device):
    sae.eval()
    face_sum = torch.zeros(sae.n_dirs, device=device)
    outside_sum = torch.zeros_like(face_sum)
    face_cells = outside_cells = 0
    for sid in ids:
        residual, mask = load_face_tokens(sid, boxes, residual_dir, device)
        codes = sae.encode(residual)
        face_sum += codes[mask].sum(dim=0)
        outside_sum += codes[~mask].sum(dim=0)
        face_cells += int(mask.sum())
        outside_cells += int((~mask).sum())
    face_mean = face_sum / max(face_cells, 1)
    outside_mean = outside_sum / max(outside_cells, 1)
    contrast = face_mean - outside_mean
    order = torch.argsort(contrast, descending=True)
    sae.train()
    return [
        {
            "rank": rank + 1,
            "feature": int(index),
            "mean_face_activation": float(face_mean[index]),
            "mean_outside_activation": float(outside_mean[index]),
            "face_minus_outside": float(contrast[index]),
        }
        for rank, index in enumerate(order[:20])
    ]


@torch.no_grad()
def render_test_heatmaps(sae, ranked, test_rows, boxes, residual_dir, output, device):
    features = [r["feature"] for r in ranked[:3]]
    selected_rows = {}
    ordered_rows = sorted(
        test_rows,
        key=lambda row: (row["race_prompt"], row["gender_prompt"], int(row["seed"])),
    )
    for row in ordered_rows:
        key = (row["race_prompt"], row["gender_prompt"])
        selected_rows.setdefault(key, row)
    heat_dir = output / "heatmaps"
    heat_dir.mkdir()
    tile_size = (160, 184)
    tile_w, tile_h = tile_size
    tiles = []
    for row in selected_rows.values():
        sid = row["sample_id"]
        residual, _ = load_face_tokens(sid, boxes, residual_dir, device)
        codes = sae.encode(residual)
        image_path = ROOT / "backprop" / row["image"]
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        box = boxes[sid]
        outputs = [("original", original.copy())]
        for feature in features:
            overlay = make_feature_overlay(original, codes[..., feature])
            outputs.append((f"feature {feature}", overlay))
        rectangle = (box[0] * 512, box[1] * 512, box[2] * 512, box[3] * 512)
        for _, image in outputs:
            ImageDraw.Draw(image).rectangle(rectangle, outline="#00e5ff", width=3)
        group = f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"
        sheet = make_heatmap_sheet(group, outputs, tile_size)
        sheet.save(heat_dir / f"{sid}.jpg", quality=90)
        tiles.append((group, sheet))
    contact = Image.new("RGB", (tile_w * 4, tile_h * math.ceil(len(tiles) / 4)), "white")
    for i, (_, tile) in enumerate(tiles):
        contact.paste(tile, ((i % 4) * tile_w * 4, (i // 4) * tile_h))
    contact.save(heat_dir / "heldout_face_feature_maps.jpg", quality=90)


def main():
    args = parse_args()
    boxes, eligible, split = load_training_data(args)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAE fine-tuning")
    device = torch.device("cuda")

    base = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).eval()
    sae = SparseAutoencoder.load_from_disk(str(args.checkpoint)).float().to(device).train()
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, eps=1e-8)
    checkpoint_hash = sha256(args.checkpoint / "state_dict.pth")
    args.output.mkdir(parents=True)

    history = []
    best_val = float("inf")
    best_epoch = None
    snapshots_dir = args.output / "checkpoints"
    snapshots_dir.mkdir()
    # Epoch 0 is the pretrained SAE before any updates.
    sae.save_to_disk(str(snapshots_dir / "epoch_000"))
    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        order = list(split["train"])
        random.shuffle(order)
        train_loss = train_epoch(
            sae,
            optimizer,
            order,
            boxes,
            args.residuals,
            device,
            epoch,
            unit_norm_decoder_,
            unit_norm_decoder_grad_adjustment_,
        )

        val_metrics, _ = evaluate(
            sae, split["validation"], boxes, args.residuals, device
        )
        row = {"epoch": epoch, "train_loss_mean": train_loss, **val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["face_mse"] < best_val:
            best_val = val_metrics["face_mse"]
            best_epoch = epoch
            sae.save_to_disk(str(args.output / "sae"))
        if args.snapshot_interval and epoch % args.snapshot_interval == 0:
            save_snapshot(sae, snapshots_dir, epoch)

    # Always retain the final checkpoint, including when the epoch count is
    # not divisible by the snapshot interval.
    if args.epochs % max(args.snapshot_interval, 1) != 0:
        save_snapshot(sae, snapshots_dir, args.epochs)

    # Compare the untouched baseline and selected fine-tuned SAE on held-out data.
    selected = (
        SparseAutoencoder.load_from_disk(str(args.output / "sae"))
        .float()
        .to(device)
        .eval()
    )
    results = evaluate_splits(base, selected, split, boxes, args.residuals, device)
    write_json(args.output / "training_history.json", history)
    write_json(args.output / "heldout_reconstruction.json", results)
    feature_ranking = rank_face_features(selected, split["validation"], boxes, args.residuals, device)
    write_feature_ranking(args.output / "face_localized_features.csv", feature_ranking)
    manifest_by_id = {r["sample_id"]: r for r in eligible}
    test_rows = [manifest_by_id[sample_id] for sample_id in split["test"]]
    render_test_heatmaps(
        selected,
        feature_ranking,
        test_rows,
        boxes,
        args.residuals,
        args.output,
        device,
    )
    settings = build_run_settings(
        args, checkpoint_hash, split, best_epoch, snapshots_dir
    )
    write_json(args.output / "settings.json", settings)
    print(
        json.dumps({"best_epoch": best_epoch, "heldout_reconstruction": results}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
