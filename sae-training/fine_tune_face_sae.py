"""Fine-tune a copy of the pretrained SAE on activation cells inside face boxes.

This does not train C2C or the image generator. C2C object-detection boxes are
used only as spatial masks and remain noisy, provisional labels.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESIDUALS = ROOT / "sae-training/residuals"
DEFAULT_MANIFEST = ROOT / "backprop/manifest.csv"
DEFAULT_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
DEFAULT_CHECKPOINT = ROOT / "sdxl-unbox/checkpoints/unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
BLOCK = "unet.down_blocks.2.attentions.1"


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
        if row["source"] != "objects" or row["label"].strip().lower() != "human face":
            continue
        box = [float(row[k]) / 512.0 for k in ("x1", "y1", "x2", "y2")]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result[row["sample_id"]] = box
    return result


def cell_mask(box, height, width):
    x1, y1, x2, y2 = box
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    return ((ys[:, None] >= y1) & (ys[:, None] <= y2) &
            (xs[None, :] >= x1) & (xs[None, :] <= x2))


def load_face_tokens(sample_id, boxes, residual_dir, device):
    residual = torch.from_numpy(np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False).astype(np.float32))
    h, w, d = residual.shape
    mask = cell_mask(boxes[sample_id], h, w)
    return residual.to(device), mask.to(device)


def reconstruction(sae, flat):
    rec, _ = sae(flat)
    return rec


def normalized_mse(recon, target):
    mean = target.mean(dim=0, keepdim=True)
    return F.mse_loss(recon, target) / F.mse_loss(mean.expand_as(target), target).clamp_min(1e-8)


@torch.no_grad()
def evaluate(sae, ids, boxes, residual_dir, device):
    sae.eval()
    sums = {"face_sse": 0.0, "face_n": 0, "outside_sse": 0.0, "outside_n": 0,
            "all_sse": 0.0, "all_n": 0}
    for sid in ids:
        residual, mask = load_face_tokens(sid, boxes, residual_dir, device)
        flat = residual.reshape(-1, residual.shape[-1])
        rec = reconstruction(sae, flat).reshape_as(residual)
        err = (rec - residual).square().sum(dim=-1)
        outside = ~mask
        for name, selected in (("face", mask), ("outside", outside), ("all", torch.ones_like(mask))):
            sums[f"{name}_sse"] += float(err[selected].sum())
            sums[f"{name}_n"] += int(selected.sum()) * residual.shape[-1]
        del residual, mask, flat, rec, err
    sae.train()
    return {f"{area}_mse": sums[f"{area}_sse"] / max(sums[f"{area}_n"], 1)
            for area in ("face", "outside", "all")}, sums


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
    return [{"rank": rank + 1, "feature": int(idx), "mean_face_activation": float(face_mean[idx]),
             "mean_outside_activation": float(outside_mean[idx]), "face_minus_outside": float(contrast[idx])}
            for rank, idx in enumerate(order[:20])]


@torch.no_grad()
def render_test_heatmaps(sae, ranked, test_rows, boxes, residual_dir, output, device):
    from PIL import Image, ImageDraw

    features = [r["feature"] for r in ranked[:3]]
    selected_rows = {}
    for row in sorted(test_rows, key=lambda r: (r["race_prompt"], r["gender_prompt"], int(r["seed"]))):
        selected_rows.setdefault((row["race_prompt"], row["gender_prompt"]), row)
    heat_dir = output / "heatmaps"
    heat_dir.mkdir()
    tile_w, tile_h = 160, 184
    tiles = []
    for (_, _), row in selected_rows.items():
        sid = row["sample_id"]
        residual, _ = load_face_tokens(sid, boxes, residual_dir, device)
        codes = sae.encode(residual)
        image_path = ROOT / "backprop" / row["image"]
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        box = boxes[sid]
        outputs = [("original", original.copy())]
        for feature in features:
            values = codes[..., feature].float().cpu().numpy()
            vmax = max(float(values.max()), 1e-6)
            mask_img = Image.fromarray(np.uint8(np.clip(values / vmax, 0, 1) * 255)).resize(original.size)
            overlay = Image.composite(Image.new("RGB", original.size, "red"), original, mask_img.point(lambda v: int(v * .55)))
            outputs.append((f"feature {feature}", overlay))
        for _, im in outputs:
            draw = ImageDraw.Draw(im)
            draw.rectangle((box[0] * 512, box[1] * 512, box[2] * 512, box[3] * 512), outline="#00e5ff", width=3)
        group = f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"
        sheet = Image.new("RGB", (tile_w * len(outputs), tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for col, (label, im) in enumerate(outputs):
            x = col * tile_w
            sheet.paste(im.resize((tile_w, tile_w)), (x, 20))
            draw.text((x + 3, 3), f"{group}: {label}", fill="black")
        sheet.save(heat_dir / f"{sid}.jpg", quality=90)
        tiles.append((group, sheet))
    contact = Image.new("RGB", (tile_w * 4, tile_h * math.ceil(len(tiles) / 4)), "white")
    for i, (_, tile) in enumerate(tiles):
        contact.paste(tile, ((i % 4) * tile_w * 4, (i // 4) * tile_h))
    contact.save(heat_dir / "heldout_face_feature_maps.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/face_sae_001")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot-interval", type=int, default=4,
                        help="Save SAE checkpoints every N epochs for image comparisons; 0 disables periodic snapshots")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh --output: {args.output}")
    if args.epochs < 1 or args.lr <= 0 or args.snapshot_interval < 0:
        parser.error("epochs and learning rate must be positive; snapshot interval cannot be negative")
    rows = read_csv(args.manifest)
    boxes = face_boxes(args.boxes)
    eligible = [r for r in rows if r["sample_id"] in boxes and (args.residuals / f"{r['sample_id']}.npy").is_file()]
    split = {name: [r["sample_id"] for r in eligible if r["split"] == name]
             for name in ("train", "validation", "test")}
    if not all(split.values()):
        raise ValueError("Need face-boxed train, validation, and test samples")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAE fine-tuning")
    device = torch.device("cuda")

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from SAE import SparseAutoencoder
    from SAE.sae import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_

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
    for epoch in range(args.epochs):
        sae.train()
        order = list(split["train"])
        random.shuffle(order)
        losses = []
        for step, sid in enumerate(order, 1):
            residual, mask = load_face_tokens(sid, boxes, args.residuals, device)
            face = residual[mask]
            optimizer.zero_grad(set_to_none=True)
            rec, info = sae(face)
            mse_scale = 1.0 / (face.var(unbiased=False).clamp_min(1e-8))
            main_loss = F.mse_loss(rec, face) * mse_scale
            aux_rec = sae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
            aux_target = face - rec.detach() + sae.pre_bias.detach()
            aux_loss = normalized_mse(aux_rec, aux_target).nan_to_num(0.0)
            loss = main_loss + (1.0 / 32.0) * aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, image {sid}")
            loss.backward()
            unit_norm_decoder_grad_adjustment_(sae)
            optimizer.step()
            with torch.no_grad():
                unit_norm_decoder_(sae)
            losses.append(float(loss.detach()))
            if step % 120 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(order)} loss={losses[-1]:.5f}", flush=True)
            del residual, mask, face, rec, info, aux_rec, aux_target, main_loss, aux_loss, loss

        val_metrics, _ = evaluate(sae, split["validation"], boxes, args.residuals, device)
        row = {"epoch": epoch + 1, "train_loss_mean": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["face_mse"] < best_val:
            best_val = val_metrics["face_mse"]
            best_epoch = epoch + 1
            sae.save_to_disk(str(args.output / "sae"))
        completed_epoch = epoch + 1
        if args.snapshot_interval and completed_epoch % args.snapshot_interval == 0:
            sae.save_to_disk(str(snapshots_dir / f"epoch_{completed_epoch:03d}"))

    # Always retain the final checkpoint, including when the epoch count is
    # not divisible by the snapshot interval.
    if args.epochs % max(args.snapshot_interval, 1) != 0:
        sae.save_to_disk(str(snapshots_dir / f"epoch_{args.epochs:03d}"))

    # Compare the untouched baseline and selected fine-tuned SAE on held-out data.
    selected = SparseAutoencoder.load_from_disk(str(args.output / "sae")).float().to(device).eval()
    results = {"validation": {}, "test": {}}
    for name in ("validation", "test"):
        base_metrics, base_counts = evaluate(base, split[name], boxes, args.residuals, device)
        tuned_metrics, tuned_counts = evaluate(selected, split[name], boxes, args.residuals, device)
        results[name] = {"images": len(split[name]), "original_sae": base_metrics,
                         "face_fine_tuned_sae": tuned_metrics,
                         "token_counts": {"original": base_counts, "fine_tuned": tuned_counts}}
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "heldout_reconstruction.json").write_text(json.dumps(results, indent=2) + "\n")
    feature_ranking = rank_face_features(selected, split["validation"], boxes, args.residuals, device)
    with (args.output / "face_localized_features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_ranking[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(feature_ranking)
    manifest_by_id = {r["sample_id"]: r for r in eligible}
    render_test_heatmaps(selected, feature_ranking, [manifest_by_id[sid] for sid in split["test"]],
                         boxes, args.residuals, args.output, device)
    settings = {
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
        "snapshot_epochs": sorted(int(p.name.removeprefix("epoch_")) for p in snapshots_dir.iterdir()),
        "learning_rate": args.lr,
        "auxk_coefficient": 1.0 / 32.0,
        "seed": args.seed,
        "eligible_images_by_split": {k: len(v) for k, v in split.items()},
        "best_validation_epoch": best_epoch,
        "selection_metric": "validation face-region reconstruction MSE",
        "interpretation": "Exploratory face-region specialization; detector boxes are broad and unreviewed.",
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "heldout_reconstruction": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
