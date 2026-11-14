"""Apply the feature 138 alpha sweep to saved images from the biased dataset."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


FEATURE = 138
FACTORS = [-100, -80, -60, -50, -40, -30, -20, -10, -5, 0,
           5, 10, 20, 30, 40, 50, 60, 80, 100]
BLOCK = "unet.down_blocks.2.attentions.1"
DEFAULT_GROUPS = [
    "black_man", "black_woman", "white_man", "white_woman",
    "east_asian_man", "east_asian_woman", "middle_eastern_man", "middle_eastern_woman",
    "indian_man", "indian_woman", "hispanic_man", "hispanic_woman",
]


def find_project_root():
    explicit = os.environ.get("SAE_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for start in (Path(__file__).resolve(), Path.cwd()):
        for parent in (start, *start.parents):
            if (parent / "data/datasets/biased/images").is_dir() and (parent / "sdxl-unbox").is_dir():
                return parent
    raise FileNotFoundError("Set SAE_PROJECT_ROOT to the workspace containing data/datasets/biased and sdxl-unbox.")


def slug(value):
    value = int(value)
    return f"m{abs(value)}" if value < 0 else f"p{value}"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                        help="biased dataset image groups (defaults to all 12 race/gender groups)")
    parser.add_argument("--samples-per-group", type=int, default=20,
                        help="number of lowest-seed images per group (default: 20, matching other feature runs)")
    parser.add_argument("--alphas", nargs="+", type=int, default=FACTORS,
                        help="alpha values to generate (default: the full saved sweep)")
    parser.add_argument("--output", default="sae-activation-tuning/tuning_activations/feature_138/run_biased_002",
                        help="output directory, relative to the project root unless absolute")
    args = parser.parse_args()
    alphas = list(dict.fromkeys(args.alphas))
    if not alphas:
        parser.error("at least one alpha value is required")
    root = find_project_root()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    manifest_path = root / "backprop/manifest.csv"
    dataset_root = root / "data/datasets/biased/images"
    with manifest_path.open(newline="") as f:
        manifest = list(csv.DictReader(f))
    expected_groups = {f"{r}_{g}" for r in ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
                       for g in ("man", "woman")}
    if not set(args.groups) <= expected_groups:
        raise ValueError(f"Unknown groups: {sorted(set(args.groups) - expected_groups)}")
    rows_by_group = {}
    for group in args.groups:
        rows = [r for r in manifest if f"{r['race_prompt'].replace(' ', '_')}_{r['gender_prompt']}" == group]
        rows.sort(key=lambda r: int(r["seed"]))
        if len(rows) < args.samples_per_group or args.samples_per_group < 1:
            raise ValueError(f"Expected at least {args.samples_per_group} images in {group}; got {len(rows)} rows")
        rows = rows[:args.samples_per_group]
        if any(r["prompt"] != f"a picture of a {r['race_prompt']} {r['gender_prompt']}" for r in rows):
            raise ValueError(f"Expected simple-prompt biased images in {group}")
        for r in rows:
            source = dataset_root / Path(r["image"]).relative_to("images")
            if not source.is_file():
                raise FileNotFoundError(source)
        rows_by_group[group] = rows

    for folder in ("images", "activation_graphs", "seed_sheets", "post_activation_overlays",
                   "intervention_footprint_overlays", "signed_coefficient_overlays"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "sdxl-unbox"))
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings_in = json.loads((root / "backprop/generation_settings.json").read_text())
    checkpoint = root / "sdxl-unbox/checkpoints" / (
        f"{BLOCK}_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
    )
    torch.set_num_threads(4)
    load = dict(torch_dtype=torch.float16, cache_dir=str(root / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", vae=vae, variant="fp16", **load
    )
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
    sae.requires_grad_(False)

    def hook_for(alpha, captured):
        @torch.no_grad()
        def hook(module, inputs, output):
            original = output[0]
            residual = (original - inputs[0]).permute(0, 2, 3, 1).float()
            codes = sae.encode(residual)
            coefficient = codes[..., FEATURE:FEATURE + 1]
            change = coefficient * (alpha - 1.0)
            delta = change @ sae.decoder.weight[:, FEATURE:FEATURE + 1].T
            edited = original + delta.permute(0, 3, 1, 2).to(original)
            post = sae.encode((edited - inputs[0]).permute(0, 2, 3, 1).float())
            if not torch.isfinite(edited).all() or not torch.isfinite(post).all():
                raise FloatingPointError(f"Non-finite activations at alpha {alpha}")
            captured.update(
                pre=coefficient[0, ..., 0].float().cpu().numpy(),
                post=post[0, ..., FEATURE].float().cpu().numpy(),
                scaled=(coefficient[0, ..., 0] * alpha).float().cpu().numpy(),
                pre_mean=float(coefficient.mean().item()),
                scaled_mean=float((coefficient * alpha).mean().item()),
                scaled_abs_mean=float((coefficient * alpha).abs().mean().item()),
                change_mean=float(change.abs().mean().item()),
                post_mean=float(post[..., FEATURE].mean().item()),
                active_fraction=float((post[..., FEATURE] > 0).float().mean().item()),
            )
            return (edited,) + output[1:]
        return hook

    map_keys, pre_maps, post_maps, scaled_maps, footprint_maps = [], [], [], [], []
    metric_rows, review_rows = [], []
    for group in args.groups:
        group_maps = {k: [] for k in ("pre", "post", "scaled", "footprint")}
        for row in rows_by_group[group]:
            seed = int(row["seed"])
            source = dataset_root / Path(row["image"]).relative_to("images")
            baseline = output / "images" / group / "baseline" / f"seed{seed:03d}.png"
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, baseline)
            base_rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
            key_pre = None
            for alpha in alphas:
                captured = {}
                # Match the source dataset's independent CUDA generators for noise and scheduler.
                noise_rng = torch.Generator("cuda").manual_seed(seed)
                scheduler_rng = torch.Generator("cuda").manual_seed(seed)
                latents = torch.randn((1, 4, 64, 64), generator=noise_rng, device="cuda", dtype=torch.float16)
                result = pipe.run_with_hooks(
                    prompt=row["prompt"], latents=latents, generator=scheduler_rng,
                    num_inference_steps=settings_in["steps"], guidance_scale=settings_in["guidance_scale"],
                    height=settings_in["height"], width=settings_in["width"],
                    position_hook_dict={BLOCK: hook_for(alpha, captured)},
                )
                image = result.images[0].convert("RGB")
                image_path = output / "images" / group / f"seed{seed:03d}_alpha_{slug(alpha)}.png"
                image.save(image_path)
                mae = float(np.abs(np.asarray(image, dtype=np.float32) - base_rgb).mean())
                if key_pre is None:
                    key_pre = captured["pre"]
                group_maps["post"].append(captured["post"])
                group_maps["scaled"].append(captured["scaled"])
                group_maps["footprint"].append(np.abs((alpha - 1) * captured["pre"]))
                metric_rows.append({
                    "group": group, "seed": seed, "feature": FEATURE, "alpha": alpha,
                    "prompt": row["prompt"], "source_image": str(source.relative_to(root)),
                    "pre_mean_activation": captured["pre_mean"],
                    "mean_scaled_coefficient": captured["scaled_mean"],
                    "mean_abs_scaled_coefficient": captured["scaled_abs_mean"],
                    "mean_abs_intervention_coefficient_change": captured["change_mean"],
                    "post_reencoded_mean_activation": captured["post_mean"],
                    "post_reencoded_active_fraction": captured["active_fraction"],
                    "pixel_mae_vs_original": mae,
                })
                review_rows.append({
                    "group": group, "seed": seed, "feature": FEATURE, "alpha": alpha,
                    "source_image": str(source.relative_to(root)), "visible_attribute_notes": "",
                    "face_or_hair_damage": "", "other_artifacts": "", "notes": "",
                })
            group_maps["pre"].append(key_pre)
            map_keys.append(f"{group}|{seed}")
            print(f"Completed {group} seed={seed}; {len(metric_rows)} alpha results", flush=True)
        pre_maps.extend(group_maps["pre"])
        post_maps.extend(np.asarray(group_maps["post"]).reshape(len(rows_by_group[group]), len(alphas), *group_maps["post"][0].shape))
        scaled_maps.extend(np.asarray(group_maps["scaled"]).reshape(len(rows_by_group[group]), len(alphas), *group_maps["scaled"][0].shape))
        footprint_maps.extend(np.asarray(group_maps["footprint"]).reshape(len(rows_by_group[group]), len(alphas), *group_maps["footprint"][0].shape))

    metric_rows.sort(key=lambda r: (args.groups.index(r["group"]), int(r["seed"]), int(r["alpha"])))
    review_rows.sort(key=lambda r: (args.groups.index(r["group"]), int(r["seed"]), int(r["alpha"])))
    write_csv(output / "metrics.csv", metric_rows)
    write_csv(output / "visual_review.csv", review_rows)
    np.savez_compressed(
        output / "activation_maps.npz",
        pre_maps=np.asarray(pre_maps, dtype=np.float32),
        post_reencoded_maps=np.asarray(post_maps, dtype=np.float32),
        intervention_footprint_maps=np.asarray(footprint_maps, dtype=np.float32),
        scaled_coefficient_maps=np.asarray(scaled_maps, dtype=np.float32),
        map_keys=np.asarray(map_keys), multipliers=np.asarray(alphas),
    )
    settings_out = {
        "feature": FEATURE,
        "source_dataset": "data/datasets/biased (simple-prompt set)",
        "source_manifest": "backprop/manifest.csv",
        "groups": args.groups,
        "samples_per_group": args.samples_per_group,
        "prompt_template": settings_in["prompt_template"],
        "alphas": alphas,
        "intervention": "block output plus (alpha - 1) times the original SAE coefficient times its decoder direction; alpha +1 is the no-op",
        "block": BLOCK,
        "checkpoint": str(checkpoint.relative_to(root)),
        "model": settings_in["model"], "vae": settings_in["vae"],
        "steps": settings_in["steps"], "guidance_scale": settings_in["guidance_scale"],
        "height": settings_in["height"], "width": settings_in["width"],
        "scheduler": settings_in["scheduler"], "timestep_spacing": settings_in["timestep_spacing"],
        "seed_replay": "The original biased-dataset seed and prompt are replayed for each alpha; original dataset PNG is retained for the original tile and pixel MAE reference.",
        "sae_weights_sha256": settings_in["sae_weights_sha256"],
        "run": output.name,
    }
    (output / "settings.json").write_text(json.dumps(settings_out, indent=2) + "\n")
    print(f"Saved {len(metric_rows)} alpha results in {output}; now run render_signed_visuals.py", flush=True)


if __name__ == "__main__":
    main()
