"""Render fixed held-out SDXL portraits through SAE checkpoints from a training run."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "backprop"
BLOCK = "unet.down_blocks.2.attentions.1"


def reconstruction_hook(sae):
    import torch

    @torch.no_grad()
    def hook(module, inputs, output):
        original = output[0]
        residual = (original - inputs[0]).permute(0, 2, 3, 1).float()
        codes = sae.encode(residual)
        reconstructed = sae.decoder(codes) + sae.pre_bias
        updated = inputs[0] + reconstructed.permute(0, 3, 1, 2).to(original)
        return (updated,) + output[1:]
    return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Training run directory containing checkpoints/epoch_NNN")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="backprop data directory with manifest.csv and images/")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/reconstructions")
    parser.add_argument("--seeds-per-cell", type=int, default=1,
                        help="Number of test images per race/gender prompt cell")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.seeds_per_cell < 1:
        parser.error("seeds-per-cell must be positive")
    run = args.run.resolve()
    output = (args.output or run / "reconstructions").resolve()
    if output.exists():
        raise ValueError(f"Output exists; choose a fresh --output: {output}")

    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings = json.loads((run / "settings.json").read_text())
    if settings["block"] != BLOCK:
        raise ValueError(f"This renderer expects {BLOCK}, found {settings['block']}")
    checkpoint_root = run / "checkpoints"
    snapshots = sorted(checkpoint_root.glob("epoch_[0-9][0-9][0-9]"))
    if not snapshots or snapshots[0].name != "epoch_000":
        raise ValueError("Run must contain checkpoints/epoch_000 and epoch snapshots")
    with (args.data / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for gender in ("man", "woman"):
        for race in ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic"):
            cell = [r for r in rows if r["split"] == "test" and r["race_prompt"].replace(" ", "_") == race
                    and r["gender_prompt"] == gender]
            selected.extend(cell[:args.seeds_per_cell])
    if not selected:
        raise ValueError("No matching held-out test images in manifest")

    load = dict(torch_dtype=torch.float16, cache_dir=str(args.cache_dir),
                local_files_only=not args.allow_download)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)

    output.mkdir(parents=True)
    labels = [f"epoch {int(p.name.split('_')[1])}" for p in snapshots]
    (output / "settings.json").write_text(json.dumps({
        "run": str(run), "data": str(args.data), "sample_ids": [r["sample_id"] for r in selected],
        "epochs": [int(p.name.split("_")[1]) for p in snapshots], "block": BLOCK,
        "steps": 1, "guidance_scale": 0.0,
        "note": "Original stored image plus generated SAE-reconstruction outputs using fixed prompts and seeds."
    }, indent=2) + "\n")

    all_rows = []
    for row in selected:
        seed = int(row["seed"])
        def gen_args():
            latents = torch.randn((1, 4, 64, 64),
                                  generator=torch.Generator("cuda").manual_seed(seed),
                                  device="cuda", dtype=torch.float16)
            return dict(prompt=row["prompt"], latents=latents,
                        generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=1,
                        guidance_scale=0.0, height=512, width=512)
        with Image.open(args.data / row["image"]) as source:
            original = source.convert("RGB")
        tiles = [("original", original.copy())]
        for snapshot, label in zip(snapshots, labels):
            sae = SparseAutoencoder.load_from_disk(str(snapshot)).float().cuda().eval()
            sae.requires_grad_(False)
            image = pipe.run_with_hooks(**gen_args(), position_hook_dict={BLOCK: reconstruction_hook(sae)}).images[0]
            tiles.append((label, image))
            del sae
            torch.cuda.empty_cache()
        directory = output / row["sample_id"]
        directory.mkdir()
        tile_w, tile_h = 192, 216
        sheet = Image.new("RGB", (tile_w * len(tiles), tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(tiles):
            image.save(directory / f"{label.replace(' ', '_')}.png")
            x = i * tile_w
            sheet.paste(image.resize((tile_w, tile_w)), (x, 24))
            draw.text((x + 4, 5), label, fill="black")
        sheet.save(directory / "comparison.jpg", quality=92)
        all_rows.append((row["sample_id"], sheet))
        print(f"Rendered {row['sample_id']}", flush=True)

    columns = 2
    page_w, page_h = 192 * len(snapshots + [None]), 216
    overview = Image.new("RGB", (page_w * columns, page_h * ((len(all_rows) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(overview)
    for i, (sample_id, sheet) in enumerate(all_rows):
        x, y = (i % columns) * page_w, (i // columns) * page_h
        overview.paste(sheet, (x, y))
        draw.text((x + 3, y + page_h - 14), sample_id, fill="black")
    overview.save(output / "all_samples.jpg", quality=90)
    print(f"Saved reconstruction comparisons to {output}", flush=True)


if __name__ == "__main__":
    main()
