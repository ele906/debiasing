"""Ablate face-SAE feature 3725 on matched, existing portrait seeds."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(ROOT / "sdxl-unbox/scripts"))
from SAE import SparseAutoencoder
from SDLens import HookedStableDiffusionXLPipeline
from sae_experiment import intervention
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "sae-training/runs/face_parts_sae_001/independent_test_seed300_309")
    p.add_argument("--sae", type=Path, default=ROOT / "sae-training/runs/face_sae_020/sae")
    p.add_argument("--output", type=Path, default=ROOT / "sae-training/runs/face_sae_020/feature_3725_ablation_001")
    p.add_argument("--feature", type=int, default=3725)
    args = p.parse_args()
    data, sae_dir, out = args.data.resolve(), args.sae.resolve(), args.output.resolve()
    manifest = list(csv.DictReader((data / "manifest.csv").open(newline="")))
    women = {"black_woman", "white_woman", "east_asian_woman", "middle_eastern_woman", "indian_woman", "hispanic_woman"}
    rows = [r for r in manifest if r["sample_id"].rsplit("_seed", 1)[0] in women]
    rows.sort(key=lambda r: (int(r["seed"]), r["sample_id"]))
    if not rows:
        raise RuntimeError(f"No women prompt-group images in {data}")
    if not torch.cuda.is_available():
        raise RuntimeError("This intervention requires a CUDA GPU")
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        "experiment": "face SAE feature 3725 attenuation on matched existing images",
        "feature": args.feature,
        "sae_path": str(sae_dir),
        "sae_weights_sha256": hashlib.sha256((sae_dir / "state_dict.pth").read_bytes()).hexdigest(),
        "source_data": str(data), "samples": len(rows),
        "prompt_groups": sorted(women), "seeds": sorted({int(r["seed"]) for r in rows}),
        "strengths": [1.0, 0.5, 0.0],
        "strength_definition": "1 is unchanged; 0.5 halves selected feature contribution; 0 removes selected feature contribution",
        "activation_block": "unet.down_blocks.2.attentions.1",
        "model": "stabilityai/sdxl-turbo", "steps": 1, "guidance_scale": 0.0,
        "interpretation_limit": "Group association only; images are not labeled or controlled for bindis, earrings, or headwear.",
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    source_settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    if hashlib.sha256((Path(source_settings["sae_checkpoint"]) / "state_dict.pth").read_bytes()).hexdigest() != source_settings["sae_weights_sha256"]:
        raise RuntimeError("Pretrained diffusion SAE hash mismatch")
    torch.set_num_threads(4)
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(source_settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(source_settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(sae_dir)).float().cuda().eval()
    sae.requires_grad_(False)

    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)
    metrics = []
    image_lookup = {}
    for i, row in enumerate(rows, 1):
        sid, seed = row["sample_id"], int(row["seed"])
        latents = torch.randn((1, 4, 64, 64), generator=torch.Generator("cuda").manual_seed(seed), device="cuda", dtype=torch.float16)
        kwargs = dict(prompt=row["prompt"], latents=latents,
                      generator=torch.Generator("cuda").manual_seed(seed),
                      num_inference_steps=1, guidance_scale=0.0, height=512, width=512)
        with torch.no_grad():
            baseline = pipe.pipe(**kwargs).images[0]
            original_path = data / row["image"]
            original = Image.open(original_path).convert("RGB")
            replay_error = float(np.abs(np.asarray(baseline).astype(np.float32) - np.asarray(original).astype(np.float32)).mean())
            if replay_error > 0.05:
                raise RuntimeError(f"Source replay mismatch for {sid}: pixel MAE {replay_error:.6f}")
            image_lookup[(sid, 1.0)] = original
            for strength in (0.5, 0.0):
                edited = pipe.run_with_hooks(**kwargs, position_hook_dict={
                    source_settings["block"]: intervention(sae, [args.feature], strength)
                }).images[0]
                delta = float(np.abs(np.asarray(edited).astype(np.float32) - np.asarray(baseline).astype(np.float32)).mean())
                image_path = images_dir / f"{sid}_feature{args.feature}_strength{strength:g}.png"
                edited.save(image_path)
                image_lookup[(sid, strength)] = edited
                metrics.append({"sample_id": sid, "race_prompt": row["race_prompt"], "seed": seed,
                                "feature": args.feature, "strength": strength,
                                "pixel_mae_vs_replay": delta, "replay_mae_vs_source": replay_error,
                                "image": str(image_path.relative_to(out))})
        print(f"{i}/{len(rows)} {sid}", flush=True)

    write_csv(out / "metrics.csv", metrics)
    groups = ["black_woman", "white_woman", "east_asian_woman", "middle_eastern_woman", "indian_woman", "hispanic_woman"]
    seeds = sorted({int(r["seed"]) for r in rows})
    thumb = 96
    sheet = Image.new("RGB", (len(groups) * 3 * thumb, 30 + len(seeds) * (thumb + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, group in enumerate(groups):
        for k, label in enumerate(("original", "half", "zero")):
            draw.text((j * 3 * thumb + k * thumb + 2, 7), f"{group.replace('_woman','')} {label}", fill="black")
    for ri, seed in enumerate(seeds):
        for gi, group in enumerate(groups):
            sid = f"{group}_seed{seed:03d}"
            for ci, strength in enumerate((1.0, 0.5, 0.0)):
                image = image_lookup[(sid, strength)].resize((thumb, thumb), Image.Resampling.LANCZOS)
                x, y = gi * 3 * thumb + ci * thumb, 28 + ri * (thumb + 22)
                sheet.paste(image, (x, y))
            draw.text((gi * 3 * thumb + 2, 28 + ri * (thumb + 22) + thumb + 2), f"seed {seed}", fill="black")
    sheet.save(out / "comparison_contact_sheet.jpg", quality=92)

    report = ["# Face SAE feature 3725 ablation diagnostic", "",
              f"This run compares the original replay with half-strength and zero-strength feature {args.feature} on {len(rows)} existing women-group portraits (seeds {min(seeds)}–{max(seeds)}). The SAE and diffusion model are frozen.", "",
              "Zero strength removes only the selected feature's decoder contribution at the SAE hook. Half strength attenuates that contribution by 50%. The original is the strength-1 control.", "",
              "## Mean image change", "",
              "Pixel MAE is measured against the unchanged replay on a 0–255 scale. It measures visual change, not whether any specific attribute was removed.", "",
              "| Prompt group | Half strength MAE | Zero strength MAE |", "|---|---:|---:|"]
    for group in groups:
        half = np.mean([m["pixel_mae_vs_replay"] for m in metrics if m["race_prompt"].replace(" ", "_") + "_woman" == group and m["strength"] == 0.5])
        zero = np.mean([m["pixel_mae_vs_replay"] for m in metrics if m["race_prompt"].replace(" ", "_") + "_woman" == group and m["strength"] == 0.0])
        report.append(f"| {group.replace('_woman','').replace('_',' ')} | {half:.3f} | {zero:.3f} |")
    report += ["", "![Original, half-strength, and zero-strength comparison](comparison_contact_sheet.jpg)", "",
               "## Interpretation", "",
               "This is a diagnostic of feature 3725's causal effect on these generated outputs. The source dataset does not label or control bindis, earrings, or headwear, so an image change cannot be called bindi or earring removal. Any change may involve other upper-face or styling details. Visual review is needed alongside the pixel-change summary.", ""]
    (out / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({"output": str(out), "samples": len(rows), "feature": args.feature,
                      "contact_sheet": str(out / "comparison_contact_sheet.jpg")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
