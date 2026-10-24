"""Generate a small paired bindi/earring control pilot for visual verification."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
import diffusers
import numpy as np
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw
from SAE import SparseAutoencoder
from SDLens import HookedStableDiffusionXLPipeline

MODEL = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"
BLOCK = "unet.down_blocks.2.attentions.1"
SAE_DIR = ROOT / "sae-training/runs/face_sae_020/sae"
SEEDS = range(310, 316)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/face_sae_020/bindi_earring_pilot_002")
    p.add_argument("--seed-start", type=int, default=316)
    p.add_argument("--seed-count", type=int, default=6)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--guidance", type=float, default=1.5)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = "studio portrait photograph of an Indian woman, facing camera, neutral expression, hair pulled back, both ears visible, plain gray background, even studio lighting"
    rows = []
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    for seed in seeds:
        for bindi in (0, 1):
            for earrings in (0, 1):
                forehead = "a single small bright red bindi centered between her eyebrows" if bindi else "a completely bare, unmarked forehead with no dots"
                ears = "small gold earrings clearly visible on both ears" if earrings else "no earrings; both earlobes are bare"
                sid = f"indian_woman_bindi{bindi}_earrings{earrings}_seed{seed}"
                negative = []
                if not bindi:
                    negative.extend(("bindi", "red dot on forehead", "forehead mark", "forehead jewelry"))
                if not earrings:
                    negative.extend(("earrings", "ear jewelry"))
                rows.append({"sample_id": sid, "seed": seed, "bindi_prompt": bindi,
                             "earrings_prompt": earrings, "prompt": f"{base}; {forehead}; {ears}",
                             "negative_prompt": ", ".join(negative),
                             "image": f"images/{sid}.png", "residual": f"residuals/{sid}.npy"})
    settings = {
        "experiment": "paired bindi and earring prompt pilot with negative prompts; visual verification before SAE comparison",
        "model": MODEL, "vae": VAE, "activation_block": BLOCK,
        "sae_checkpoint": str(SAE_DIR), "seed_range": [min(seeds), max(seeds) - 1],
        "images": len(rows), "factorial_conditions": {"bindi": [0, 1], "earrings": [0, 1]},
        "steps": args.steps, "guidance_scale": args.guidance, "height": 512, "width": 512,
        "negative_prompt": "feature-specific negative conditioning when target condition is off",
        "labels": "prompt conditions only; manual visual verification is required",
        "torch_version": torch.__version__, "diffusers_version": diffusers.__version__,
    }
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Existing pilot settings differ; use a fresh output directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    write_csv(out / "manifest.csv", rows)

    if not torch.cuda.is_available():
        raise RuntimeError("This generation pilot requires CUDA")
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(VAE, **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(SAE_DIR)).float().cuda().eval()
    sae.requires_grad_(False)

    start, activations = time.monotonic(), []
    image_dir, residual_dir = out / "images", out / "residuals"
    image_dir.mkdir(exist_ok=True)
    residual_dir.mkdir(exist_ok=True)
    for i, row in enumerate(rows, 1):
        seed = int(row["seed"])
        latents = torch.randn((1, 4, 64, 64), generator=torch.Generator("cuda").manual_seed(seed), device="cuda", dtype=torch.float16)
        with torch.no_grad():
            result, cache = pipe.run_with_cache(
                prompt=row["prompt"], latents=latents,
                generator=torch.Generator("cuda").manual_seed(seed),
                negative_prompt=row["negative_prompt"] or None,
                num_inference_steps=args.steps, guidance_scale=args.guidance, height=512, width=512,
                positions_to_cache=[BLOCK], save_input=True)
            residual = (cache["output"][BLOCK] - cache["input"][BLOCK])[:, -1].permute(0, 2, 3, 1).float()
            code = sae.encode(residual)
            feature_mean = code.mean(dim=(0, 1, 2)).cpu().numpy()
            feature_max = code.amax(dim=(0, 1, 2)).cpu().numpy()
        result.images[0].save(out / row["image"])
        np.save(out / row["residual"], residual[0].cpu().numpy(), allow_pickle=False)
        activations.append({"sample_id": row["sample_id"], "mean": feature_mean, "maximum": feature_max})
        del cache, code, residual
        if i == 1 or i % 8 == 0:
            print(f"{i}/{len(rows)} images generated", flush=True)

    np.savez_compressed(out / "face_sae_activations.npz",
                        sample_id=np.array([r["sample_id"] for r in rows]),
                        seed=np.array([r["seed"] for r in rows]),
                        bindi_prompt=np.array([r["bindi_prompt"] for r in rows]),
                        earrings_prompt=np.array([r["earrings_prompt"] for r in rows]),
                        X_mean=np.stack([a["mean"] for a in activations]),
                        X_max=np.stack([a["maximum"] for a in activations]))

    sheet = Image.new("RGB", (4 * 320, (args.seed_count + 1) * 345), "white")
    draw = ImageDraw.Draw(sheet)
    conditions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for col, (bindi, earrings) in enumerate(conditions):
        draw.text((col * 320 + 5, 6), f"bindi {bindi} | earrings {earrings} (prompt target)", fill="black")
    for ri, seed in enumerate(seeds):
        for ci, (bindi, earrings) in enumerate(conditions):
            sid = f"indian_woman_bindi{bindi}_earrings{earrings}_seed{seed}"
            image = Image.open(out / "images" / f"{sid}.png").convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
            x, y = ci * 320, 28 + ri * 345
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + 322), f"seed {seed}", fill="black")
    sheet.save(out / "pilot_contact_sheet.jpg", quality=93)
    print(json.dumps({"images": len(rows), "seconds": time.monotonic() - start,
                      "contact_sheet": str(out / "pilot_contact_sheet.jpg")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
