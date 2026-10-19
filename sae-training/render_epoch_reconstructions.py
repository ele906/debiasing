"""Compare fixed-seed images and SAE feature maps across training epochs."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "backprop"
BLOCK = "unet.down_blocks.2.attentions.1"
MODEL = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")

sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SAE import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run containing checkpoint snapshots")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Dataset directory with manifest.csv and images/")
    parser.add_argument("--output", type=Path,
                        help="Output directory (default: RUN/reconstructions)")
    parser.add_argument("--seeds-per-cell", type=int, default=1,
                        help="Held-out test images to render for each prompt group")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed sample sheets in an existing output directory")
    args = parser.parse_args()
    if args.seeds_per_cell < 1:
        parser.error("--seeds-per-cell must be positive")
    args.run = args.run.resolve()
    args.data = args.data.resolve()
    args.output = (args.output or args.run / "reconstructions").resolve()
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output already exists; choose a new directory: {args.output}")
    return args


def read_snapshots(run_dir):
    checkpoint_dir = run_dir / "checkpoints"
    snapshots = sorted(checkpoint_dir.glob("epoch_[0-9][0-9][0-9]"))
    if not snapshots or snapshots[0].name != "epoch_000":
        raise ValueError("Run must contain checkpoints/epoch_000 and epoch snapshots")
    return snapshots


def read_test_samples(data_dir, seeds_per_cell):
    with (data_dir / "manifest.csv").open(newline="") as manifest:
        rows = list(csv.DictReader(manifest))

    selected = []
    for gender in GENDERS:
        for race in RACES:
            group = [
                row for row in rows
                if row["split"] == "test"
                and row["race_prompt"].replace(" ", "_") == race
                and row["gender_prompt"] == gender
            ]
            selected.extend(group[:seeds_per_cell])
    if not selected:
        raise ValueError("No matching held-out test images in manifest")
    return selected


def read_top_features(run_dir, limit=3):
    path = run_dir / "face_localized_features.csv"
    with path.open(newline="") as feature_file:
        rows = list(csv.DictReader(feature_file))
    features = [int(row["feature"]) for row in rows[:limit]]
    if not features:
        raise ValueError(f"No ranked features found in {path}")
    return features


def load_pipeline(cache_dir, allow_download):
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler

    from SDLens import HookedStableDiffusionXLPipeline

    load_options = {
        "torch_dtype": torch.float16,
        "cache_dir": str(cache_dir),
        "local_files_only": not allow_download,
    }
    vae = AutoencoderKL.from_pretrained(VAE, **load_options)
    pipeline = HookedStableDiffusionXLPipeline.from_pretrained(
        MODEL, vae=vae, variant="fp16", **load_options
    )
    pipeline.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipeline.pipe.to("cuda")
    pipeline.pipe.set_progress_bar_config(disable=True)
    return pipeline


def reconstruction_hook(sae, activation_cache):
    @torch.no_grad()
    def hook(_module, inputs, output):
        residual = (output[0] - inputs[0]).permute(0, 2, 3, 1).float()
        codes = sae.encode(residual)
        activation_cache.append(codes[-1].detach().cpu())
        reconstruction = sae.decoder(codes) + sae.pre_bias
        updated = inputs[0] + reconstruction.permute(0, 3, 1, 2).to(output[0])
        return (updated,) + output[1:]

    return hook


def generation_args(row):
    seed = int(row["seed"])
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=torch.Generator("cuda").manual_seed(seed),
        device="cuda",
        dtype=torch.float16,
    )
    return {
        "prompt": row["prompt"],
        "latents": latents,
        "generator": torch.Generator("cuda").manual_seed(seed),
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
        "height": 512,
        "width": 512,
    }


def jet_overlay(image, activations):
    values = activations.float().cpu().numpy()
    values = np.clip(values / max(float(values.max()), 1e-6), 0, 1)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    rgb = np.uint8(np.stack((red, green, blue), axis=-1) * 255)
    alpha = np.uint8(values * 180)
    rgba = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
    rgba = rgba.resize(image.size, resample=Image.Resampling.NEAREST)
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def render_sample(row, snapshots, features, pipeline, data_dir):
    with Image.open(data_dir / row["image"]) as source:
        original = source.convert("RGB")

    epochs = []
    for snapshot in snapshots:
        sae = SparseAutoencoder.load_from_disk(str(snapshot)).float().cuda().eval()
        sae.requires_grad_(False)
        activation_cache = []
        image = pipeline.run_with_hooks(
            **generation_args(row),
            position_hook_dict={BLOCK: reconstruction_hook(sae, activation_cache)},
        ).images[0]
        if not activation_cache:
            raise RuntimeError(f"No activations captured at {BLOCK} for {snapshot.name}")

        feature_images = {
            feature: jet_overlay(image, activation_cache[0][..., feature])
            for feature in features
        }
        epoch = int(snapshot.name.split("_")[1])
        epochs.append({"epoch": epoch, "reconstruction": image, "features": feature_images})
        del sae
        torch.cuda.empty_cache()

    return {"row": row, "original": original, "epochs": epochs}


def image_paths(sample_dir, epochs, features):
    paths = [sample_dir / "original.png"]
    for epoch in epochs:
        paths.append(sample_dir / f"epoch_{epoch:03d}_reconstruction.png")
        paths.extend(
            sample_dir / f"epoch_{epoch:03d}_feature_{feature}.png"
            for feature in features
        )
    return paths


def save_sample_comparison(sample, output_dir, features):
    row = sample["row"]
    sample_dir = output_dir / row["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample["original"].save(sample_dir / "original.png")

    tile_width = 160
    tile_size = 144
    row_height = 166
    header_height = 52
    label_width = 76
    columns = ["reconstruction", *(f"feature {feature}" for feature in features)]
    sheet = Image.new(
        "RGB",
        (label_width + len(columns) * tile_width,
         header_height + len(sample["epochs"]) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), row["sample_id"], fill="black")
    for column, label in enumerate(columns):
        draw.text((label_width + column * tile_width + 4, 31), label, fill="black")

    for row_index, result in enumerate(sample["epochs"]):
        epoch = result["epoch"]
        y = header_height + row_index * row_height
        draw.text((4, y + 8), f"epoch {epoch}", fill="black")
        reconstruction = result["reconstruction"]
        reconstruction.save(sample_dir / f"epoch_{epoch:03d}_reconstruction.png")
        images = [reconstruction]
        for feature in features:
            overlay = result["features"][feature]
            overlay.save(sample_dir / f"epoch_{epoch:03d}_feature_{feature}.png")
            images.append(overlay)
        for column, image in enumerate(images):
            x = label_width + column * tile_width
            sheet.paste(image.resize((tile_size, tile_size)), (x + 8, y + 18))

    sheet.save(sample_dir / "contact_sheet.jpg", quality=92)
    sheet.save(sample_dir / "comparison.jpg", quality=92)
    return sheet


def load_saved_sample(row, snapshots, features, output_dir):
    sample_dir = output_dir / row["sample_id"]
    epochs = [int(snapshot.name.split("_")[1]) for snapshot in snapshots]
    expected = image_paths(sample_dir, epochs, features)
    if not all(path.is_file() for path in expected):
        return None

    with Image.open(sample_dir / "original.png") as source:
        original = source.convert("RGB")
    epoch_results = []
    for epoch in epochs:
        with Image.open(sample_dir / f"epoch_{epoch:03d}_reconstruction.png") as source:
            reconstruction = source.convert("RGB")
        feature_images = {}
        for feature in features:
            path = sample_dir / f"epoch_{epoch:03d}_feature_{feature}.png"
            with Image.open(path) as source:
                feature_images[feature] = source.convert("RGB")
        epoch_results.append({
            "epoch": epoch,
            "reconstruction": reconstruction,
            "features": feature_images,
        })
    return {"row": row, "original": original, "epochs": epoch_results}


def save_gender_overview(samples, output_path):
    image_size = 224
    cell_width = 240
    row_height = 260
    header_height = 42
    label_width = 84
    panel_width = label_width + cell_width
    rows_per_race = max(
        sum(sample["row"]["race_prompt"].replace(" ", "_") == race
            and sample["row"]["gender_prompt"] == gender for sample in samples)
        for race in RACES
        for gender in GENDERS
    )
    overview = Image.new(
        "RGB",
        (panel_width * len(GENDERS), header_height + row_height * len(RACES) * rows_per_race),
        "white",
    )
    draw = ImageDraw.Draw(overview)

    for gender_index, gender in enumerate(GENDERS):
        panel_x = gender_index * panel_width
        draw.text((panel_x + 4, 4), "MEN" if gender == "man" else "WOMEN", fill="black")
        samples_by_race = {
            race: sorted(
                (sample for sample in samples
                 if sample["row"]["gender_prompt"] == gender
                 and sample["row"]["race_prompt"].replace(" ", "_") == race),
                key=lambda sample: int(sample["row"]["seed"]),
            )
            for race in RACES
        }
        for race_index, race in enumerate(RACES):
            race_samples = samples_by_race[race]
            for sample_index in range(rows_per_race):
                y = header_height + (race_index * rows_per_race + sample_index) * row_height
                draw.text((panel_x + 3, y + 6), race, fill="black")
                if sample_index >= len(race_samples):
                    continue
                sample = race_samples[sample_index]
                row = sample["row"]
                draw.text((panel_x + 3, y + 24), f"seed {row['seed']}", fill="black")
                image = sample["original"].resize((image_size, image_size))
                overview.paste(image, (panel_x + label_width + 8, y + 18))

    overview.save(output_path, quality=92)


def main():
    args = parse_args()
    settings = json.loads((args.run / "settings.json").read_text())
    if settings["block"] != BLOCK:
        raise ValueError(f"Expected block {BLOCK}; found {settings['block']}")

    snapshots = read_snapshots(args.run)
    features = read_top_features(args.run)
    samples = read_test_samples(args.data, args.seeds_per_cell)
    args.output.mkdir(parents=True, exist_ok=args.resume)

    saved_samples = {
        row["sample_id"]: load_saved_sample(row, snapshots, features, args.output)
        for row in samples
    } if args.resume else {}
    needs_rendering = any(saved_samples.get(row["sample_id"]) is None for row in samples)
    pipeline = load_pipeline(args.cache_dir, args.allow_download) if needs_rendering else None

    metadata = {
        "run": str(args.run),
        "data": str(args.data),
        "sample_ids": [row["sample_id"] for row in samples],
        "epochs": [int(snapshot.name.split("_")[1]) for snapshot in snapshots],
        "heatmap_features": features,
        "feature_selection": "Top features by validation mean activation inside minus outside face boxes",
        "block": BLOCK,
        "steps": 1,
        "guidance_scale": 0.0,
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rendered_samples = []
    for row in samples:
        saved = saved_samples.get(row["sample_id"])
        if saved is None:
            rendered = render_sample(row, snapshots, features, pipeline, args.data)
            saved = rendered
            print(f"Rendered {row['sample_id']}", flush=True)
        else:
            print(f"Reused {row['sample_id']}", flush=True)
        save_sample_comparison(saved, args.output, features)
        rendered_samples.append(saved)

    save_gender_overview(rendered_samples, args.output / "all_samples.jpg")
    print(f"Saved comparison overview to {args.output / 'all_samples.jpg'}", flush=True)


if __name__ == "__main__":
    main()
