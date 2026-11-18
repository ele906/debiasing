"""Generate a resumable up.0.0 sweep shard with paired clean controls."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from PIL import Image

from capture import BLOCK, MODEL, ROOT, VAE
from discover import RUN, group, write_csv
from SAE import SparseAutoencoder
from SDLens import HookedStableDiffusionXLPipeline

PILOT = [1, -30, -7, 0, 3, 10, 30]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "full"])
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    features = json.loads((RUN / "discovery/top10.json").read_text())["features"]
    historical = json.loads((ROOT / "layers/mid0_stability/top10_submission.json").read_text())
    if args.mode == "full":
        gate = json.loads((RUN / "pilot/validation.json").read_text())
        if not gate["passed"] or gate["checkpoint_sha256"] != digest(RUN / "training/sae/state_dict.pth"):
            raise RuntimeError("Pilot validation required for this checkpoint")
        alphas = [1] + [a for a in historical["alphas"] if a != 1]
    else:
        features = features[:3]
        alphas = PILOT
    feature = features[args.index]
    out = RUN / args.mode / f"feature_{feature}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = RUN / "training/sae"
    import importlib.metadata as metadata
    manifest = ROOT / "backprop/manifest.csv"
    settings_path = ROOT / "backprop/generation_settings.json"
    settings = json.loads(settings_path.read_text())
    config = dict(feature=feature, block=BLOCK, alphas=alphas, seeds=list(range(30, 35)),
                  checkpoint_sha256=digest(checkpoint / "state_dict.pth"),
                  checkpoint_files={p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()},
                  manifest_sha256=digest(manifest), generation_settings_sha256=digest(settings_path),
                  mode=args.mode, code_sha256=digest(Path(__file__)),
                  software={name: metadata.version(name) for name in ("torch", "diffusers", "transformers")},
                  device=torch.cuda.get_device_name())
    provenance = out / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != config:
        raise ValueError(f"Shard provenance changed: {out}")
    write_json(provenance, config)
    prior = RUN / "pilot" / out.name
    if args.mode == "full" and (prior / "manifest.csv").is_file():
        prior_config = json.loads((prior / "provenance.json").read_text())
        comparable = set(config) - {"mode", "alphas"}
        if any(config[key] != prior_config.get(key) for key in comparable):
            raise ValueError(f"Pilot reuse provenance changed: {prior}")
    else:
        prior = None

    def reuse(destination, expected=None):
        if prior is None:
            return False
        source = prior / destination.relative_to(out)
        if not source.is_file():
            return False
        if expected is not None and digest(source) != expected:
            raise ValueError(f"Pilot output failed integrity check: {source}")
        temp = destination.with_suffix(".reuse")
        temp.unlink(missing_ok=True)
        os.link(source, temp)
        temp.replace(destination)
        return True
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if 30 <= int(row["seed"]) <= 34]
    if len({(group(row), int(row["seed"])) for row in rows}) != 60 or len(rows) != 60:
        raise ValueError("Expected 60 matched prompt and seed pairs")

    options = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(VAE, **options)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, variant="fp16", **options)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()

    def generate(row, hook=None):
        seed = int(row["seed"])
        noise = torch.Generator("cuda").manual_seed(seed)
        scheduler = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=noise, device="cuda", dtype=torch.float16)
        options = dict(prompt=row["prompt"], latents=latents, generator=scheduler,
                       num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                       height=settings["height"], width=settings["width"])
        result = pipe.pipe(**options) if hook is None else pipe.run_with_hooks(
            **options, position_hook_dict={BLOCK: hook})
        return result.images[0].convert("RGB")

    records = []
    for row in rows:
        label = group(row)
        seed = int(row["seed"])
        sample = out / label / f"seed{seed:03d}"
        sample.mkdir(parents=True, exist_ok=True)
        baseline_file = sample / "baseline.png"
        if not baseline_file.exists() and not reuse(baseline_file):
            baseline = generate(row)
            baseline.save(sample / "baseline.tmp.png")
            (sample / "baseline.tmp.png").replace(baseline_file)
        with Image.open(baseline_file) as image:
            baseline = np.asarray(image.convert("RGB"), dtype=np.float32)
        original_path = ROOT / "data/datasets/biased/images" / Path(row["image"]).relative_to("images")
        with Image.open(original_path) as image:
            original_pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        baseline_hash = digest(baseline_file)
        mid_run = Path(historical["output_root"])
        mid_feature = historical["features"][0]
        mid_control = mid_run / f"feature_{mid_feature}/images/{label}/seed{seed:03d}_alpha_p1.png"
        with Image.open(mid_control) as reference:
            matched_mid0 = bool(np.array_equal(baseline, np.asarray(reference.convert("RGB"))))
        for alpha in alphas:
            image_path = sample / f"alpha_{alpha}.png"
            metrics_path = image_path.with_suffix(".json")
            maps_path = image_path.with_suffix(".npz")
            if metrics_path.exists():
                result = json.loads(metrics_path.read_text())
                if (result["image_sha256"] != digest(image_path)
                    or result["maps_sha256"] != digest(maps_path)
                    or result["baseline_sha256"] != baseline_hash):
                    raise ValueError(f"Saved condition failed integrity check: {metrics_path}")
                records.append(result)
                continue
            if prior is not None:
                old_metrics = prior / metrics_path.relative_to(out)
                if old_metrics.is_file():
                    result = json.loads(old_metrics.read_text())
                    if result["baseline_sha256"] != baseline_hash:
                        raise ValueError(f"Pilot baseline changed: {old_metrics}")
                    if not (reuse(image_path, result["image_sha256"]) and reuse(maps_path, result["maps_sha256"])):
                        raise FileNotFoundError(f"Incomplete pilot output: {old_metrics}")
                    result["image"] = str(image_path.relative_to(RUN))
                    write_json(metrics_path, result)
                    records.append(result)
                    continue
            measured = {}
            maps = {}

            def hook(module, inputs, output):
                original = output[0]
                if original.shape != (1, 1280, 16, 16) or measured:
                    raise RuntimeError("Unexpected up.0.0 hook shape or invocation count")
                residual = (original - inputs[0]).permute(0, 2, 3, 1).float()
                codes = sae.encode(residual)
                coefficient = codes[..., feature:feature + 1]
                delta = (coefficient * (alpha - 1.0)) @ sae.decoder.weight[:, feature:feature + 1].T
                edited = original + delta.permute(0, 3, 1, 2).to(original)
                post = sae.encode((edited - inputs[0]).permute(0, 2, 3, 1).float())
                applied = edited.float() - original.float()
                if not torch.isfinite(edited).all() or not torch.isfinite(post).all():
                    raise FloatingPointError("Nonfinite activation")
                l2 = applied.norm().item()
                zero = not bool(coefficient.any())
                intended = codes.clone()
                intended[..., feature] *= alpha
                reconstruction_error = (sae.decoder(post - intended)).norm().item()
                measured.update(reencoded_vs_intended_decoder_l2=reconstruction_error,
                                delta_fro_applied=l2, delta_fro_precast=delta.norm().item(),
                                relative_delta_block_fro=l2 / max(original.float().norm().item(), 1e-12),
                                relative_delta_residual_fro=l2 / max(residual.norm().item(), 1e-12),
                                delta_rms=l2 / original.numel() ** 0.5,
                                pre_mean_activation=coefficient.mean().item(), zero_activation=zero,
                                pre_active_fraction=(coefficient > 0).float().mean().item(),
                                post_active_fraction=(post[..., feature] > 0).float().mean().item(),
                                newly_active_fraction=((post > 0) & (codes == 0)).float().mean().item(),
                                decoder_norm=sae.decoder.weight[:, feature].norm().item())
                maps.update(pre=coefficient[0, ..., 0].cpu().numpy(),
                            post=post[0, ..., feature].cpu().numpy(),
                            applied_l2=applied[0].norm(dim=0).cpu().numpy())
                return (edited,) + output[1:]

            image = generate(row, hook)
            if not measured:
                raise RuntimeError("Requested attention hook did not run")
            pixels = np.asarray(image, dtype=np.float32)
            matches = bool(np.array_equal(pixels, baseline))
            if (alpha == 1 or measured["zero_activation"]) and (
                not matches or measured["delta_fro_applied"] != 0
            ):
                raise ValueError(f"No-op failed for {label}, seed {seed}, alpha {alpha}")
            if args.mode == "pilot" and alpha in (1, -7):
                measured.clear()
                replay = generate(row, hook)
                if not np.array_equal(pixels, np.asarray(replay)):
                    raise ValueError(f"Intervention replay failed for {label}, seed {seed}, alpha {alpha}")
            image.save(image_path)
            np.savez_compressed(maps_path, **maps)
            result = dict(group=label, seed=seed, feature=feature, alpha=alpha, prompt=row["prompt"],
                          image=str(image_path.relative_to(RUN)), image_sha256=digest(image_path),
                          maps_sha256=digest(maps_path), baseline_sha256=baseline_hash,
                          matches_clean_baseline=matches,
                          clean_matches_mid0_control=matched_mid0,
                          clean_matches_historical_baseline=bool(np.array_equal(baseline, original_pixels)),
                          pixel_mae_vs_original=float(np.abs(pixels - original_pixels).mean()),
                          pixel_mae_vs_clean=float(np.abs(pixels - baseline).mean()), **measured)
            write_json(metrics_path, result)
            records.append(result)
        print(f"Feature {feature}: {label}, seed {seed} complete", flush=True)
    write_csv(out / "manifest.csv", records)


if __name__ == "__main__":
    main()
