"""Capture and encode matching SDXL Turbo samples with layer-specific SAEs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SDLens import HookedStableDiffusionXLPipeline
from SAE import SparseAutoencoder

BLOCKS = {
    "unet.down_blocks.2.attentions.1": "unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final",
    "unet.mid_block.attentions.0": "unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/final",
    "unet.up_blocks.0.attentions.0": "unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/final",
    "unet.up_blocks.0.attentions.1": "unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final",
}
MODEL = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def face_boxes(path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for row in read_csv(path):
        if row["label"].strip().lower() != "human face":
            continue
        box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        if box[2] > box[0] and box[3] > box[1]:
            result.setdefault(row["sample_id"], []).append(box)
    return result


def face_mask(boxes, height: int, width: int) -> np.ndarray:
    xs = (np.arange(width) + 0.5) * (512 / width)
    ys = (np.arange(height) + 0.5) * (512 / height)
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "sae-training/runs/face_parts_sae_001/test_extension")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "layers/runs/turbo_layer_depth_001")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a fresh directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    chunk_dir = output / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    rows = [r for r in read_csv(data / "manifest.csv")
            if args.seed_start <= int(r["seed"]) < args.seed_start + args.seed_count]
    rows.sort(key=lambda r: (r["gender_prompt"], r["race_prompt"], int(r["seed"])))
    if not rows:
        raise RuntimeError("No manifest rows match the requested seed range")
    boxes = face_boxes(data / "annotations/bounding_boxes.csv")
    with (output / "manifest.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This capture requires a GPU")
    torch.set_grad_enabled(False)
    load = dict(torch_dtype=torch.float16, cache_dir=str(args.cache_dir.resolve()),
                local_files_only=True)
    vae = AutoencoderKL.from_pretrained(VAE, **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae,
                                                           variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    saes = {}
    hashes = {}
    for block, checkpoint in BLOCKS.items():
        checkpoint_path = ROOT / "sdxl-unbox/checkpoints" / checkpoint
        weight_path = checkpoint_path / "state_dict.pth"
        hashes[block] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        sae = SparseAutoencoder.load_from_disk(str(checkpoint_path)).float().cuda().eval()
        sae.requires_grad_(False)
        if sae.d_model != 1280 or sae.n_dirs != 5120:
            raise ValueError(f"Unexpected SAE dimensions for {block}: {sae.d_model}, {sae.n_dirs}")
        saes[block] = sae

    def encode_chunk(chunk_rows: list[dict[str, str]], chunk_id: int) -> Path:
        result = {"sample_id": [], "race_prompt": [], "gender_prompt": [],
                  "seed": [], "image": []}
        per_block = {block: {key: [] for key in
                             ("face_mean", "face_max", "face_active", "global_mean")}
                     for block in BLOCKS}
        for local_i, row in enumerate(chunk_rows, 1):
            sid, seed = row["sample_id"], int(row["seed"])
            noise_rng = torch.Generator("cuda").manual_seed(seed)
            scheduler_rng = torch.Generator("cuda").manual_seed(seed)
            latents = torch.randn((1, 4, 64, 64), generator=noise_rng,
                                  device="cuda", dtype=torch.float16)
            _, cache = pipe.run_with_cache(
                prompt=row["prompt"], latents=latents, generator=scheduler_rng,
                num_inference_steps=1, guidance_scale=0.0, height=512, width=512,
                positions_to_cache=list(BLOCKS), save_input=True)
            result["sample_id"].append(sid)
            result["race_prompt"].append(row["race_prompt"])
            result["gender_prompt"].append(row["gender_prompt"])
            result["seed"].append(seed)
            result["image"].append(row["image"])
            for block, sae in saes.items():
                residual = (cache["output"][block] - cache["input"][block])[:, -1]
                residual = residual.permute(0, 2, 3, 1).float()
                z = sae.encode(residual)[0].float()
                h, w = z.shape[:2]
                mask = face_mask(boxes.get(sid, []), h, w)
                if mask.any():
                    roi = z[torch.from_numpy(mask).to(z.device)]
                    face_mean = roi.mean(0)
                    face_max = roi.amax(0)
                    face_active = (roi > 0).float().mean(0)
                else:
                    face_mean = torch.full((sae.n_dirs,), float("nan"), device=z.device)
                    face_max = face_mean.clone()
                    face_active = face_mean.clone()
                per_block[block]["face_mean"].append(face_mean.cpu().numpy())
                per_block[block]["face_max"].append(face_max.cpu().numpy())
                per_block[block]["face_active"].append(face_active.cpu().numpy())
                per_block[block]["global_mean"].append(z.mean((0, 1)).cpu().numpy())
                del residual, z
            del cache, latents
            if local_i == 1 or local_i % 5 == 0:
                print(f"chunk {chunk_id}: {local_i}/{len(chunk_rows)} samples encoded",
                      flush=True)
        arrays = {key: np.asarray(value) for key, value in result.items()}
        for block, values in per_block.items():
            prefix = "b" + str(list(BLOCKS).index(block))
            for key, value in values.items():
                arrays[f"{prefix}_{key}"] = np.stack(value).astype(np.float32)
        path = chunk_dir / f"chunk_{chunk_id:04d}.npz"
        temp = chunk_dir / f"chunk_{chunk_id:04d}.tmp.npz"
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
        return path

    chunk_paths = []
    for chunk_id, start in enumerate(range(0, len(rows), args.chunk_size)):
        path = chunk_dir / f"chunk_{chunk_id:04d}.npz"
        chunk_rows = rows[start:start + args.chunk_size]
        if not path.exists():
            encode_chunk(chunk_rows, chunk_id)
        chunk_paths.append(path)
    chunks = [np.load(path, allow_pickle=False) for path in chunk_paths]
    merged = {key: np.concatenate([c[key] for c in chunks], axis=0)
              for key in chunks[0].files}
    for i, block in enumerate(BLOCKS):
        prefix = f"b{i}_"
        np.savez_compressed(output / f"layer_{block.replace('.', '_')}.npz",
                            sample_id=merged["sample_id"], race_prompt=merged["race_prompt"],
                            gender_prompt=merged["gender_prompt"], seed=merged["seed"],
                            image=merged["image"],
                            X_face_mean=merged[prefix + "face_mean"],
                            X_face_max=merged[prefix + "face_max"],
                            X_face_active_fraction=merged[prefix + "face_active"],
                            X_global_mean=merged[prefix + "global_mean"])
    settings = {
        "experiment": "same-sample layer-specific SAE comparison on SDXL Turbo",
        "model": MODEL, "vae": VAE, "steps": 1, "guidance_scale": 0.0,
        "height": 512, "width": 512, "seeds": [args.seed_start,
            args.seed_start + args.seed_count - 1], "samples": len(rows),
        "split": "discovery seeds 100-149; held-out seeds 150-199",
        "activation_convention": "U-Net block output minus input; final cached sublayer",
        "pooling": "mean/max/active fraction inside automatic face boxes, plus global mean",
        "face_box_source": str(data / "annotations/bounding_boxes.csv"),
        "layers": list(BLOCKS), "sae_weight_sha256": hashes,
        "saes": "matching pretrained layer-specific SDXL Turbo SAEs; no fine-tuning in this run",
        "image_source": str(data / "images"), "device": torch.cuda.get_device_name(0),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps({"output": str(output), "samples": len(rows),
                      "layers": list(BLOCKS)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
