"""Render held-out portraits with SAE checkpoints from a training run."""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "backprop"
BLOCK = "unet.down_blocks.2.attentions.1"
MODEL = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"
RACES = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
GENDERS = ("man", "woman")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run containing checkpoints/epoch_NNN")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Dataset directory with manifest.csv and images/")
    parser.add_argument("--output", type=Path, help="Output directory (default: RUN/reconstructions)")
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
            group = [row for row in rows
                     if row["split"] == "test"
                     and row["race_prompt"].replace(" ", "_") == race
                     and row["gender_prompt"] == gender]
            selected.extend(group[:seeds_per_cell])

    if not selected:
        raise ValueError("No matching held-out test images in manifest")
    return selected


def load_pipeline(cache_dir, allow_download):
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
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


def reconstruction_hook(sae):
    import torch

    @torch.no_grad()
    def hook(_module, inputs, output):
        original = output[0]
        residual = (original - inputs[0]).permute(0, 2, 3, 1).float()
        codes = sae.encode(residual)
        reconstruction = sae.decoder(codes) + sae.pre_bias
        updated = inputs[0] + reconstruction.permute(0, 3, 1, 2).to(original)
        return (updated,) + output[1:]

    return hook


def generation_args(row):
    import torch

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


def render_checkpoint_images(row, snapshots, pipeline, sae_class, data_dir):
    import torch

    with Image.open(data_dir / row["image"]) as source:
        images = [("original", source.convert("RGB"))]

    for snapshot in snapshots:
        sae = sae_class.load_from_disk(str(snapshot)).float().cuda().eval()
        sae.requires_grad_(False)
        result = pipeline.run_with_hooks(
            **generation_args(row),
            position_hook_dict={BLOCK: reconstruction_hook(sae)},
        ).images[0]
        epoch = int(snapshot.name.split("_")[1])
        images.append((f"epoch {epoch}", result))
        del sae
        torch.cuda.empty_cache()

    return images


def save_comparison(sample_id, images, output_dir):
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(exist_ok=True)
    tile_width, tile_height = 192, 216
    sheet = Image.new("RGB", (tile_width * len(images), tile_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (label, image) in enumerate(images):
        image.save(sample_dir / f"{label.replace(' ', '_')}.png")
        x = index * tile_width
        sheet.paste(image.resize((tile_width, tile_width)), (x, 24))
        draw.text((x + 4, 5), label, fill="black")

    sheet.save(sample_dir / "comparison.jpg", quality=92)
    return sheet


def save_overview(comparisons, output_path):
    columns = 2
    page_width = comparisons[0][1].width
    page_height = comparisons[0][1].height + 18
    rows = (len(comparisons) + columns - 1) // columns
    overview = Image.new("RGB", (page_width * columns, page_height * rows), "white")
    draw = ImageDraw.Draw(overview)

    for index, (sample_id, sheet) in enumerate(comparisons):
        x = (index % columns) * page_width
        y = (index // columns) * page_height
        overview.paste(sheet, (x, y))
        draw.text((x + 3, y + sheet.height + 2), sample_id, fill="black")

    overview.save(output_path, quality=90)


def main():
    args = parse_args()
    settings = json.loads((args.run / "settings.json").read_text())
    if settings["block"] != BLOCK:
        raise ValueError(f"Expected block {BLOCK}; found {settings['block']}")

    snapshots = read_snapshots(args.run)
    samples = read_test_samples(args.data, args.seeds_per_cell)
    pipeline = load_pipeline(args.cache_dir, args.allow_download)

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from SAE import SparseAutoencoder

    args.output.mkdir(parents=True, exist_ok=args.resume)
    epochs = [int(snapshot.name.split("_")[1]) for snapshot in snapshots]
    run_settings = {
        "run": str(args.run),
        "data": str(args.data),
        "sample_ids": [row["sample_id"] for row in samples],
        "epochs": epochs,
        "block": BLOCK,
        "steps": 1,
        "guidance_scale": 0.0,
        "note": "Original image plus SAE-reconstructed generations using fixed prompts and seeds.",
    }
    (args.output / "settings.json").write_text(json.dumps(run_settings, indent=2) + "\n")

    comparisons = []
    for row in samples:
        sample_dir = args.output / row["sample_id"]
        comparison_path = sample_dir / "comparison.jpg"
        if comparison_path.is_file():
            with Image.open(comparison_path) as existing_sheet:
                comparisons.append((row["sample_id"], existing_sheet.convert("RGB")))
            print(f"Reused {row['sample_id']}", flush=True)
            continue
        images = render_checkpoint_images(
            row, snapshots, pipeline, SparseAutoencoder, args.data
        )
        sheet = save_comparison(row["sample_id"], images, args.output)
        comparisons.append((row["sample_id"], sheet))
        print(f"Rendered {row['sample_id']}", flush=True)

    save_overview(comparisons, args.output / "all_samples.jpg")
    print(f"Saved comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
