"""
Generation step only: sample seeds per (race, gender) category, run them through
SDXL-turbo + the down.2.1 SAE, max-pool the activations over space, and save the
resulting [N, n_dirs] feature matrix + labels to disk.

This is the expensive GPU step — run it once, then iterate on train_classify.py
(logistic regression / nearest-centroid / plots) against the cached .npz without
regenerating images every time.

Output:
  race_classifier/race_classify_features.npz  (X, y_race, y_gender, races, genders)
  race_classifier/images/<race>_<gender>/seed<seed>.png  (first N_SAVE_IMAGES_PER_CATEGORY per category)
"""
import os
os.chdir('/n/fs/goose/ReNO')

import sys
sys.path.append('/n/fs/goose/sdxl-unbox')

import re
import numpy as np
import torch
from pytorch_lightning import seed_everything

from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from SDLens import HookedStableDiffusionXLPipeline
from SAE import SparseAutoencoder

CACHE_DIR = 'hf_cache'
N_INFERENCE_STEPS = 1
DTYPE = torch.float16
DEVICE = torch.device('cuda')

CODE = 'down.2.1'
CODE_TO_BLOCK = {
    "down.2.1": "unet.down_blocks.2.attentions.1",
}
CHECKPOINT_DIR = '/n/fs/goose/sdxl-unbox/checkpoints'
FEATURES_NPZ = '/n/fs/goose/race_classifier/race_classify_features.npz'
IMAGE_DIR = '/n/fs/goose/race_classifier/images'
os.makedirs(IMAGE_DIR, exist_ok=True)

RACES = ["black", "white", "east asian", "middle eastern", "indian", "hispanic"]
GENDERS = ["man", "woman"]
CATEGORIES = [(race, gender) for gender in GENDERS for race in RACES]

PROMPT_TEMPLATE = (
    "a portrait of a {race} {gender} with a neutral expression, "
    "without any facial coverings, in a neutral white colored t-shirt against a neutral white background"
)

N_SEEDS_PER_CATEGORY = 50   # -> 600 samples total, 100 per race across both genders
N_SAVE_IMAGES_PER_CATEGORY = 5  # how many of the N_SEEDS_PER_CATEGORY samples to also save as PNGs
RNG_SEED = 1234


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


print("Loading pipeline + SAE...", flush=True)
hooked_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE, cache_dir=CACHE_DIR,
)
hooked_pipe = HookedStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", vae=hooked_vae, torch_dtype=DTYPE, variant="fp16",
    use_safetensors=True, cache_dir=CACHE_DIR,
)
hooked_pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    hooked_pipe.pipe.scheduler.config, timestep_spacing="trailing",
)
hooked_pipe.pipe = hooked_pipe.pipe.to(DEVICE, DTYPE)

sae = SparseAutoencoder.load_from_disk(
    os.path.join(CHECKPOINT_DIR, f"{CODE_TO_BLOCK[CODE]}_k10_hidden5120_auxk256_bs4096_lr0.0001", "final")
).to(DEVICE)
n_dirs = sae.n_dirs
print("n_dirs:", n_dirs, flush=True)


def get_pooled_feats(prompt, seed, save_image_path=None):
    seed_everything(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        images, cache = hooked_pipe.run_with_cache(
            prompt,
            latents=latents,
            generator=generator,
            num_inference_steps=N_INFERENCE_STEPS,
            guidance_scale=0.0,
            positions_to_cache=[CODE_TO_BLOCK[CODE]],
            save_input=True,
            save_output=True,
        )

    if save_image_path is not None:
        images.images[0].save(save_image_path)

    block = CODE_TO_BLOCK[CODE]
    diff = cache["output"][block] - cache["input"][block]
    if diff.shape[0] == 2:
        diff = diff[1].unsqueeze(0)
    diff_last = diff[:, -1].permute(0, 2, 3, 1).squeeze(0)  # [h, w, d_model]
    h, w, d_model = diff_last.shape
    with torch.no_grad():
        feats = sae.encode(diff_last.reshape(-1, d_model).float().to(DEVICE))
    feats = feats.reshape(h, w, n_dirs)
    pooled = feats.amax(dim=(0, 1)).cpu().numpy()  # [n_dirs], max-pool over space
    return pooled


print(f"Sampling {N_SEEDS_PER_CATEGORY} seeds x {len(CATEGORIES)} categories...", flush=True)
rng = np.random.default_rng(RNG_SEED)

X_list, y_race_list, y_gender_list = [], [], []
for race, gender in CATEGORIES:
    prompt = PROMPT_TEMPLATE.format(race=race, gender=gender)
    chosen_seeds = rng.choice(10000, size=N_SEEDS_PER_CATEGORY, replace=False).tolist()
    cat_dir = os.path.join(IMAGE_DIR, slugify(f"{race}_{gender}"))
    os.makedirs(cat_dir, exist_ok=True)
    for i, seed in enumerate(chosen_seeds):
        save_path = os.path.join(cat_dir, f"seed{seed}.png") if i < N_SAVE_IMAGES_PER_CATEGORY else None
        pooled = get_pooled_feats(prompt, seed, save_image_path=save_path)
        X_list.append(pooled)
        y_race_list.append(RACES.index(race))
        y_gender_list.append(GENDERS.index(gender))
    print(f"  done: {race} {gender}", flush=True)

X = np.stack(X_list)  # [N, n_dirs]
y_race = np.array(y_race_list)
y_gender = np.array(y_gender_list)
print("X shape:", X.shape, flush=True)

np.savez(
    FEATURES_NPZ,
    X=X, y_race=y_race, y_gender=y_gender, races=np.array(RACES), genders=np.array(GENDERS),
)
print(f"Saved features to {FEATURES_NPZ}")
