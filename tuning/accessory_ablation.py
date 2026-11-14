"""Same-prompt SAE ablation pilot for headscarves and earrings.

Uses existing SAE-concept candidates; it does not train the SAE or diffusion
model. Florence-2 labels are unreviewed and are only used to sample examples.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "accessory_ablation"
TARGETS = {
    # Rank-1 candidates selected using discovery data in
    # concept/sae_comparison_full/candidates.csv.
    "headscarf": (1257, "middle eastern", "woman", r"\b(head\s*scarf|headdress)\b"),
    "earrings": (4858, "east asian", "woman", r"\bearrings?\b"),
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    predictions = read_csv(ROOT / "concept/concept_predictions.csv")
    selected = []
    for name, (feature, race, gender, pattern) in TARGETS.items():
        group = [r for r in predictions if r["split"] == "validation" and
                 r["race_prompt"] == race and r["gender_prompt"] == gender]
        positive = [r for r in group if re.search(pattern, r["detected_concepts"].lower())]
        negative = [r for r in group if not re.search(pattern, r["detected_concepts"].lower())]
        if min(len(positive), len(negative)) < 4:
            raise ValueError(f"Not enough same-prompt validation examples for {name}: "
                             f"{len(positive)} positive, {len(negative)} negative")
        rng = np.random.default_rng(20261012 + feature)
        for label, pool in (("detector_positive", positive), ("detector_negative", negative)):
            for row in rng.choice(pool, 4, replace=False):
                selected.append({"concept": name, "target_feature": feature,
                                 "source_label": label,
                                 **{k: row[k] for k in ("sample_id", "split", "race_prompt",
                                                        "gender_prompt", "image", "features")}})
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "manifest.csv", selected)
    (OUT / "settings.json").write_text(json.dumps({
        "targets": {k: {"feature": v[0], "prompt_group": [v[1], v[2]], "pattern": v[3]}
                    for k, v in TARGETS.items()},
        "strengths": [-1, -0.5, 0, 0.5, 1], "samples_per_concept": 8,
        "selection": "Fixed-seed balanced sample from same-prompt validation rows; "
                     "automatic Florence-2 labels are unreviewed.",
        "candidate_selection": "Rank 1 from discovery-only full SAE-concept comparison.",
    }, indent=2) + "\n")
    print(f"Prepared {len(selected)} examples")


def run():
    import sys
    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    sys.path.insert(0, str(ROOT / "sdxl-unbox/scripts"))
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    from sae_experiment import intervention

    settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    checkpoint = Path(settings["sae_checkpoint"])
    if hashlib.sha256((checkpoint / "state_dict.pth").read_bytes()).hexdigest() != settings["sae_weights_sha256"]:
        raise ValueError("SAE checkpoint hash mismatch")
    manifest = read_csv(OUT / "manifest.csv")
    pooled = np.load(ROOT / "backprop/sae_features.npz", allow_pickle=False)
    pooled_ids = list(pooled["sample_id"].astype(str))
    pooled_mean = pooled["X_mean"]
    pooled_index = {sample_id: i for i, sample_id in enumerate(pooled_ids)}

    target_ids = {name: value[0] for name, value in TARGETS.items()}
    rng = np.random.default_rng(20261012)
    controls = {}
    for name, feature in target_ids.items():
        rows = [r for r in manifest if r["concept"] == name]
        x = pooled_mean[[pooled_index[r["sample_id"]] for r in rows]]
        mean, frequency = x.mean(0), (x > 0).mean(0)
        candidates = [j for j in range(x.shape[1]) if j not in target_ids.values()
                      and mean[j] > 0 and abs(frequency[j] - frequency[feature]) <= .25
                      and .5 * mean[feature] <= mean[j] <= 2 * mean[feature]]
        if not candidates:
            raise ValueError(f"No matched random feature for {name}")
        controls[name] = int(rng.choice(candidates))

    out_images, out_grids = OUT / "images", OUT / "grids"
    out_images.mkdir(exist_ok=True)
    out_grids.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae,
                                                            variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
    sae.requires_grad_(False)

    metrics = []
    for i, row in enumerate(manifest, 1):
        seed = int(Path(row["image"]).stem[-3:])
        kwargs = dict(prompt=f'a picture of a {row["race_prompt"]} {row["gender_prompt"]}',
                      latents=torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16,
                                          generator=torch.Generator("cuda").manual_seed(seed)),
                      generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=1,
                      guidance_scale=0., height=512, width=512)
        base = pipe.pipe(**kwargs).images[0]
        source = Image.open(row["image"]).convert("RGB")
        replay_error = float(np.abs(np.asarray(base).astype(float) - np.asarray(source)).mean())
        if replay_error != 0:
            raise RuntimeError(f"Source replay mismatch for {row['sample_id']}: {replay_error}")
        panels = []
        for role, feature in (("target", target_ids[row["concept"]]),
                              ("random_control", controls[row["concept"]])):
            tiles = []
            for alpha in (-1, -.5, 0, .5, 1):
                edited = pipe.run_with_hooks(**kwargs, position_hook_dict={
                    settings["block"]: intervention(sae, [feature], alpha)}).images[0]
                delta = float(np.abs(np.asarray(edited).astype(float) - np.asarray(base)).mean())
                if alpha == 1 and delta != 0:
                    raise RuntimeError("Strength 1 should be an exact no-op")
                path = out_images / f"{row['sample_id']}_{role}_f{feature}_a{alpha:g}.png"
                edited.save(path)
                tiles.append((f"{role} f{feature} alpha {alpha:g}", edited))
                metrics.append(dict(concept=row["concept"], sample_id=row["sample_id"],
                                    source_label=row["source_label"], feature=feature, role=role,
                                    alpha=alpha, pixel_mae=delta, replay_pixel_mae=replay_error,
                                    image=str(path.relative_to(ROOT))))
            panels.append(tiles)
        sheet = Image.new("RGB", (1280, 606), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), f"{row['concept']}: {row['sample_id']} ({row['source_label']})", fill="black")
        for y, tiles in enumerate(panels):
            for x, (label, image) in enumerate(tiles):
                draw.text((x * 256 + 3, y * 294 + 20), label, fill="black")
                sheet.paste(image.resize((256, 256)), (x * 256, y * 294 + 38))
        sheet.save(out_grids / f"{row['sample_id']}.jpg", quality=90)
        print(f"{i}/{len(manifest)} {row['sample_id']}", flush=True)

    write_csv(OUT / "metrics.csv", metrics)
    means = {}
    for concept in target_ids:
        for role in ("target", "random_control"):
            for alpha in (-1, -.5, 0, .5, 1):
                values = [float(r["pixel_mae"]) for r in metrics
                          if r["concept"] == concept and r["role"] == role and
                          float(r["alpha"]) == alpha]
                means[(concept, role, alpha)] = float(np.mean(values))
    report = ["# Headscarf and earring SAE ablation pilot", "",
              "Same-prompt validation images replayed with their original seed. The pretrained SAE and diffusion model are frozen. Strength 0 removes the selected SAE feature contribution; positive values below 1 attenuate it; negative values subtract more than the feature's measured contribution; 1 is the unchanged control. A matched random feature is included.", "",
              f"Candidate features selected from discovery data: {target_ids}. Matched random controls: {controls}.", "",
              "## Mean image change", "",
              "Mean absolute RGB pixel difference from the unchanged replay, on a 0–255 scale. This measures image change, not target removal.", "",
              "| Concept | Strength | Candidate feature | Random feature |", "|---|---:|---:|---:|"]
    for concept in target_ids:
        for alpha in (-1, -.5, 0, .5, 1):
            report.append(f"| {concept} | {alpha:g} | {means[(concept, 'target', alpha)]:.3f} | {means[(concept, 'random_control', alpha)]:.3f} |")
    report += ["", "Every saved source replay matched its original image exactly; strength 1 was an exact no-op for both candidate and random features.", "",
               "## What the pilot suggests", "",
               "In the inspected positive examples, zeroing or making the candidate negative did not clearly remove the headscarf or earrings. Negative strength changed the image more than zero for the candidate, but the matched random feature changed it even more on average. That points to a stronger activation perturbation, not reliable object control. The selected headscarf and earring features also had weak held-out concept association in the full comparison (validation AUC 0.228 and 0.407 respectively).",
               "", "Source labels come from unreviewed Florence-2 outputs; detector negatives can be missed detections. This is a small exploratory test, not evidence that these items are inherently undesirable or that removal improves fairness.", ""]
    for name in target_ids:
        report += [f"## {name}: feature {target_ids[name]}", ""]
        for row in manifest:
            if row["concept"] == name:
                report += [f"### {row['sample_id']} ({row['source_label']})", "",
                           f"![Comparison grid](grids/{row['sample_id']}.jpg)", ""]
    (OUT / "REPORT.md").write_text("\n".join(report))
    (OUT / "settings.json").write_text(json.dumps({
        "targets": target_ids, "random_controls": controls, "strengths": [-1, -.5, 0, .5, 1],
        "samples": len(manifest),
        "interpretation": "Exploratory, no training; unreviewed detector labels and image-change metrics do not establish concept removal.",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
    if not args.prepare and not args.run:
        parser.error("choose --prepare and/or --run")


if __name__ == "__main__":
    main()
