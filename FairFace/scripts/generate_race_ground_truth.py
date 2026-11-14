"""
Generates a synthetic "ground truth" test set for auditing FairFace's
accuracy in SDXL-turbo's own image domain: one prompt per race/gender
category ("a photo of a black woman's face", etc.), sampled directly with
no reward optimization (plain forward diffusion, no ReNO loop).

The prompt itself is used as a weak self-reported label -- SDXL-turbo isn't
guaranteed to render the requested category correctly, so this is not
real ground truth the way a human-labeled dataset would be. It checks a
different, narrower thing than scripts/audit_fairface_independent.py:
whether FairFace is even accurate on synthetic (non-adversarial) images
from this generator, independent of whether the ReNO reward loop later
finds a way to game it. Both checks matter and neither substitutes for
the other -- see scripts/audit_fairface_independent.py's docstring for
the adversarial-gaming concern this one does NOT cover.

Output layout:
    {save_dir}/{race_slug}_{gender_slug}/seed{N}.png

Usage:
    python scripts/generate_race_ground_truth.py --model sdxl-turbo \\
        --cache_dir /n/fs/goose/el8403/hf-cache \\
        --save_dir outputs/race_ground_truth --n_seeds 20
"""
import argparse
import logging
import os
import re

import torch
from pytorch_lightning import seed_everything

from arguments import parse_args
from models import get_latent_shape, get_model

RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]
GENDER_LABELS = ["Male", "Female"]

# Phrasing intentionally matches scripts/audit_fairface_independent.py's
# RACE_PROMPTS/GENDER_PROMPTS so results are comparable across scripts.
RACE_PHRASES = {
    "White": "white",
    "Black": "black",
    "Latino_Hispanic": "latino or hispanic",
    "East Asian": "east asian",
    "Southeast Asian": "southeast asian",
    "Indian": "indian",
    "Middle Eastern": "middle eastern",
}
GENDER_PHRASES = {"Male": "man", "Female": "woman"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    base_args = parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs/race_ground_truth")
    sweep_args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    device = torch.device("cuda")
    dtype = torch.float16 if base_args.dtype == "float16" else torch.float32

    pipe = get_model(
        base_args.model, dtype, device, base_args.cache_dir,
        base_args.memsave, base_args.cpu_offloading,
    )
    shape = get_latent_shape(pipe, base_args.model)

    for race_label in RACE_LABELS:
        for gender_label in GENDER_LABELS:
            prompt = f"a photo of a {RACE_PHRASES[race_label]} {GENDER_PHRASES[gender_label]}'s face"
            category_dir = os.path.join(
                sweep_args.save_dir, f"{slugify(race_label)}_{slugify(gender_label)}"
            )
            os.makedirs(category_dir, exist_ok=True)
            for seed in range(sweep_args.n_seeds):
                out_path = os.path.join(category_dir, f"seed{seed}.png")
                if os.path.exists(out_path):
                    logging.info(f"skipping {prompt!r} seed={seed} (already done)")
                    continue
                seed_everything(seed)
                latents = torch.randn(shape, device=device, dtype=dtype)
                generator = torch.Generator(device).manual_seed(seed)
                with torch.no_grad():
                    image = pipe.apply(
                        latents=latents,
                        prompt=prompt,
                        generator=generator,
                        num_inference_steps=base_args.n_inference_steps,
                    )
                image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                from diffusers import DiffusionPipeline
                DiffusionPipeline.numpy_to_pil(image_numpy)[0].save(out_path)
                logging.info(f"prompt={prompt!r} seed={seed} -> {out_path}")


if __name__ == "__main__":
    main()
