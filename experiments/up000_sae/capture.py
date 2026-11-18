"""Replay 1,200 prompts and save up.0.0 residual grids for SAE fine-tuning."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SDLens import HookedStableDiffusionXLPipeline

BLOCK = "unet.up_blocks.0.attentions.0"
MODEL, VAE = "stabilityai/sdxl-turbo", "madebyollin/sdxl-vae-fp16-fix"


def split(seed: int) -> str:
    if 100 <= seed <= 169:
        return "train"
    if 170 <= seed <= 184:
        return "validation"
    if 185 <= seed <= 199:
        return "test"
    raise ValueError(f"Unexpected seed: {seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "sae-training/runs/face_parts_sae_001/test_extension/manifest.csv")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "capture")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    residual_dir = output / "residuals"
    residual_dir.mkdir(exist_ok=True)
    with args.manifest.open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows = [row for row in rows if 100 <= int(row["seed"]) <= 199]
    rows.sort(key=lambda row: (row["gender_prompt"], row["race_prompt"], int(row["seed"])))
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 input rows, found {len(rows)}")
    for row in rows:
        row["split"] = split(int(row["seed"]))
    from io import StringIO
    target = StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest = output / "manifest.csv"
    if manifest.exists() and manifest.read_text() != target.getvalue():
        raise ValueError("Capture manifest changed")
    manifest.write_text(target.getvalue())

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    load = dict(torch_dtype=torch.float16, cache_dir=str(args.cache_dir.resolve()), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(VAE, **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda"); pipe.pipe.set_progress_bar_config(disable=True)
    for index, row in enumerate(rows, 1):
        path = residual_dir / f"{row['sample_id']}.npy"
        if path.exists():
            values = np.load(path, allow_pickle=False)
            if values.shape == (16, 16, 1280) and values.dtype == np.float16 and np.isfinite(values).all():
                continue
            raise ValueError(f"Invalid cached residual: {path}")
        seed = int(row["seed"])
        noise = torch.Generator("cuda").manual_seed(seed)
        scheduler = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=noise, device="cuda", dtype=torch.float16)
        _, cache = pipe.run_with_cache(prompt=row["prompt"], latents=latents, generator=scheduler,
                                       num_inference_steps=1, guidance_scale=0.0, height=512, width=512,
                                       positions_to_cache=[BLOCK], save_input=True)
        residual = (cache["output"][BLOCK] - cache["input"][BLOCK])[:, -1]
        values = residual[0].permute(1, 2, 0).float().cpu().numpy().astype(np.float16)
        if values.shape != (16, 16, 1280) or not np.isfinite(values).all():
            raise ValueError(f"Invalid residual for {row['sample_id']}: {values.shape}")
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, values, allow_pickle=False)
        temp.replace(path)
        del cache, residual, values
        if index == 1 or index % 25 == 0:
            print(f"Captured {index}/{len(rows)}", flush=True)
    checkpoint = ROOT / "sdxl-unbox/checkpoints/unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/final/state_dict.pth"
    settings = {"block": BLOCK, "model": MODEL, "vae": VAE, "images": len(rows),
                "residual_shape": [16, 16, 1280], "dtype": "float16",
                "splits": {"train": "100-169", "validation": "170-184", "test": "185-199"},
                "activation_convention": "U-Net block output minus input",
                "pretrained_sae_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
