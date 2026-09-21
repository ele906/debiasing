"""
Test whether a race label is linearly decodable from pooled SAE feature-activation
vectors, using the same generation pipeline as sae_top_features_by_race_grid.ipynb.

For each of 12 (race, gender) categories, sample N_SEEDS_PER_CATEGORY seeds, generate
an image, cache the down.2.1 block's SAE activations [h, w, n_dirs], max-pool over
space to get one n_dirs-length vector per image, and use that as a "feature activation
vector" for the sample.

Then fit a multinomial logistic regression (race -> 6 classes, gender collapsed) via
torch on a train split and evaluate held-out accuracy + confusion matrix, and print
which feature dimensions get the largest weights (to check overlap with previously
identified confound features like 138 / 3279 / 2182 / 4511).

Output: race_classifier/race_classify_results.json + printed report (captured in slurm log),
plus race_classifier/images/<race>_<gender>/seed<seed>.png for a handful of sample images
per category (N_SAVE_IMAGES_PER_CATEGORY of them) so results can be sanity-checked visually.
"""
import os
os.chdir('/n/fs/goose/ReNO')

import sys
sys.path.append('/n/fs/goose/sdxl-unbox')

import json
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
OUT_JSON = '/n/fs/goose/race_classifier/race_classify_results.json'
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
TEST_FRAC = 0.2
RNG_SEED = 1234             # different from the grid notebook's RNG_SEED=0 to get fresh seeds
EPOCHS = 300
LR = 0.05
WEIGHT_DECAY = 1e-3

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


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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

N_SAVE_IMAGES_PER_CATEGORY = 5  # how many of the N_SEEDS_PER_CATEGORY samples to also save as PNGs

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
    '/n/fs/goose/race_classifier/race_classify_features.npz',
    X=X, y_race=y_race, y_gender=y_gender, races=np.array(RACES), genders=np.array(GENDERS),
)

# --- stratified train/test split (per race x gender cell) ---
train_idx, test_idx = [], []
rng2 = np.random.default_rng(0)
for race_i in range(len(RACES)):
    for gender_i in range(len(GENDERS)):
        idx = np.where((y_race == race_i) & (y_gender == gender_i))[0]
        rng2.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_FRAC))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y_race[train_idx], y_race[test_idx]

mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
Xtr_n = (Xtr - mu) / sigma
Xte_n = (Xte - mu) / sigma

Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32, device=DEVICE)
ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
Xte_t = torch.tensor(Xte_n, dtype=torch.float32, device=DEVICE)
yte_t = torch.tensor(yte, dtype=torch.long, device=DEVICE)

n_classes = len(RACES)
W = torch.zeros(n_dirs, n_classes, device=DEVICE, requires_grad=True)
b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([W, b], lr=LR, weight_decay=WEIGHT_DECAY)

print("Training logistic regression (race, 6-class)...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    logits = Xtr_t @ W + b
    loss = torch.nn.functional.cross_entropy(logits, ytr_t)
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        with torch.no_grad():
            train_acc = (logits.argmax(-1) == ytr_t).float().mean().item()
        print(f"  epoch {epoch}: loss={loss.item():.4f} train_acc={train_acc:.3f}", flush=True)

with torch.no_grad():
    test_logits = Xte_t @ W + b
    test_pred = test_logits.argmax(-1)
    test_acc = (test_pred == yte_t).float().mean().item()

    n = n_classes
    confusion = np.zeros((n, n), dtype=int)
    for true_i, pred_i in zip(yte_t.cpu().numpy(), test_pred.cpu().numpy()):
        confusion[true_i, pred_i] += 1

    # nearest-centroid baseline for comparison
    centroids = torch.stack([Xtr_t[ytr_t == c].mean(0) for c in range(n_classes)])
    dists = torch.cdist(Xte_t, centroids)
    nc_pred = dists.argmin(-1)
    nc_acc = (nc_pred == yte_t).float().mean().item()

    # chance baseline
    chance_acc = 1.0 / n_classes

    # top-weighted features per class (which SAE dims the classifier relies on)
    top_feats_per_class = {}
    for c in range(n_classes):
        w_c = W[:, c].detach().cpu().numpy()
        top_idx = np.argsort(-np.abs(w_c))[:10]
        top_feats_per_class[RACES[c]] = [(int(i), float(w_c[i])) for i in top_idx]

print("\n=== RESULTS ===")
print(f"n_train={len(train_idx)} n_test={len(test_idx)}")
print(f"Chance accuracy: {chance_acc:.3f}")
print(f"Nearest-centroid accuracy: {nc_acc:.3f}")
print(f"Logistic regression test accuracy: {test_acc:.3f}")
print("\nConfusion matrix (rows=true, cols=pred), order:", RACES)
print(confusion)
print("\nTop-weighted SAE feature dims per race (logistic regression):")
for race, feats in top_feats_per_class.items():
    print(f"  {race}: {feats}")

results = {
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "chance_accuracy": chance_acc,
    "nearest_centroid_accuracy": nc_acc,
    "logreg_test_accuracy": test_acc,
    "races": RACES,
    "confusion_matrix": confusion.tolist(),
    "top_feats_per_class": top_feats_per_class,
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {OUT_JSON}")
