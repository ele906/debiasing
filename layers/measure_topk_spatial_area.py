"""Measure spatial SAE-map coverage for discovery-ranked features on held-out prompts.

This replays baseline samples only. It does not run feature interventions or
produce edited images.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sae-training"))
sys.path.insert(0, str(ROOT / "sdxl-unbox"))

import numpy as np
from PIL import Image

from render_signed_feature_visuals import (
    LAYERS, SHORT, attach_heldout_standardized_effects, read_csv, selected_rows,
    slug, write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "layers/runs/turbo_layer_depth_001")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 5120 or not 150 <= args.seed <= 199:
        parser.error("top-k must be 1..5120 and seed must be held-out (150..199)")

    run = args.run.resolve()
    output = (args.output or run / "results/top5_spatial_area").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selections = attach_heldout_standardized_effects(run, selected_rows(run, args.top_k))
    write_csv(output / "selected_features.csv", selections)
    settings = json.loads((run / "settings.json").read_text())
    row_by_group = {f"{r['race_prompt']} {r['gender_prompt']}": r
                    for r in read_csv(run / "manifest.csv") if int(r["seed"]) == args.seed}
    if len(row_by_group) != 12:
        raise ValueError("Expected all 12 prompt groups for the held-out seed")

    import torch
    from capture_layerwise_face_sae import BLOCKS
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    load = dict(torch_dtype=torch.float16, cache_dir=str(args.cache_dir), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae,
                                                           variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    saes = {layer: SparseAutoencoder.load_from_disk(
        str(ROOT / "sdxl-unbox/checkpoints" / BLOCKS[block])).float().cuda().eval()
        for layer, block in LAYERS.items()}
    for sae in saes.values():
        sae.requires_grad_(False)

    source_root = Path(settings["image_source"]).parent
    areas, replay_errors = [], []
    for index, (group, row) in enumerate(row_by_group.items(), 1):
        seed = int(row["seed"])
        latent_rng = torch.Generator("cuda").manual_seed(seed)
        scheduler_rng = torch.Generator("cuda").manual_seed(seed)
        result, cache = pipe.run_with_cache(
            prompt=row["prompt"],
            latents=torch.randn((1, 4, 64, 64), generator=latent_rng, device="cuda", dtype=torch.float16),
            generator=scheduler_rng, num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance_scale"], height=settings["height"], width=settings["width"],
            positions_to_cache=list(LAYERS.values()), save_input=True)
        with Image.open(source_root / row["image"]) as source:
            replay_errors.append(float(np.abs(np.asarray(result.images[0], dtype=float) -
                                              np.asarray(source.convert("RGB"), dtype=float)).max()))
        for layer, block in LAYERS.items():
            residual = (cache["output"][block] - cache["input"][block])[:, -1].permute(0, 2, 3, 1).float()
            codes = saes[layer].encode(residual)[0]
            for selection in (s for s in selections if s["layer"] == layer and s["target_group"] == group):
                feature = int(selection["feature_id_layer_specific"])
                values = codes[..., feature]
                areas.append({"layer": layer, "target_group": group, "rank": int(selection["rank"]),
                              "feature_id_layer_specific": feature, "seed": seed,
                              "grid_height": int(values.shape[0]), "grid_width": int(values.shape[1]),
                              "active_fraction": float((values > 0).float().mean().item()),
                              "heldout_standardized_contrast": selection["heldout_standardized_contrast"]})
            del codes, residual
        del cache
        print(f"replayed {index}/12: {group}", flush=True)
    if max(replay_errors) != 0:
        raise ValueError(f"Baseline replay mismatch: {max(replay_errors)}")
    write_csv(output / "spatial_area.csv", areas)

    lines = ["# Top-K selected-feature spatial coverage", "",
             f"Discovery top-K: {args.top_k}; held-out baseline seed: {args.seed}.",
             "Area is the fraction of SAE cells with `z_f > 0`, not image-pixel area or activation magnitude.", "",
             "| Layer | " + " | ".join(f"Rank {rank}" for rank in range(1, args.top_k + 1)) + " |",
             "|---|" + "|".join("---:" for _ in range(args.top_k)) + "|"]
    for layer in LAYERS:
        cells = []
        for rank in range(1, args.top_k + 1):
            values = [a["active_fraction"] for a in areas if a["layer"] == layer and a["rank"] == rank]
            cells.append(f"{np.mean(values) * 100:.1f}%")
        lines.append(f"| {SHORT[layer]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Mean held-out contrast, in feature SD", "",
                  "This uses all held-out seeds 150–199; feature selection remains discovery-only.", "",
                  "| Layer | " + " | ".join(f"Rank {rank}" for rank in range(1, args.top_k + 1)) + " |",
                  "|---|" + "|".join("---:" for _ in range(args.top_k)) + "|"])
    for layer in LAYERS:
        cells = []
        for rank in range(1, args.top_k + 1):
            values = [float(s["heldout_standardized_contrast"]) for s in selections
                      if s["layer"] == layer and int(s["rank"]) == rank]
            cells.append(f"{np.mean(values):+.2f} SD")
        lines.append(f"| {SHORT[layer]} | " + " | ".join(cells) + " |")
    lines.extend(["", "All entries average the 12 prompt groups. Feature IDs were ranked using discovery seeds 100–149 only.",
                  "The complete per-feature values are in `spatial_area.csv`.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "verification.json").write_text(json.dumps({
        "top_k": args.top_k, "seed": args.seed, "replay_samples": len(replay_errors),
        "max_source_replay_pixel_error": max(replay_errors), "maps_measured": len(areas),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
