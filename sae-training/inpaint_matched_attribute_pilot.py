"""Inpaint one bindi and earring counterfactual while preserving the same source portrait."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SOURCE_DIR = ROOT / "sae-training/runs/face_sae_020/bindi_earring_pilot_001/images"
OUT = ROOT / "sae-training/runs/face_sae_020/inpaint_attribute_pilot_004"


def bindi_center(image):
    rgb = np.asarray(image.convert("RGB"))
    region = rgb[140:260, 170:350]
    red = ((region[..., 0] > 100) &
           (region[..., 0] > region[..., 1] * 1.45) &
           (region[..., 0] > region[..., 2] * 1.35) &
           (region[..., 1] < 145))
    yy, xx = np.where(red)
    if not len(xx):
        raise RuntimeError("Could not find a red forehead mark for the edit mask")
    return int(np.median(xx) + 170), int(np.median(yy) + 140)


def soft_mask(kind, center=None, size=(512, 512)):
    scale = 4
    mask = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "bindi":
        cx, cy = center
        draw.ellipse(((cx-49)*scale, (cy-49)*scale, (cx+49)*scale, (cy+49)*scale), fill=255)
    elif kind == "earrings":
        draw.ellipse((74*scale, 285*scale, 169*scale, 417*scale), fill=255)
        draw.ellipse((361*scale, 285*scale, 455*scale, 417*scale), fill=255)
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    return mask.filter(ImageFilter.GaussianBlur(3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Inpainting requires CUDA")
    load = dict(torch_dtype=torch.float16,
                cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True,
                use_safetensors=True, variant="fp16")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL, **load)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    conditions = {
        "remove_bindi": ("photorealistic natural forehead with continuous skin texture, same person's forehead, no ornament", "bindi, red dot, forehead mark, forehead jewelry, pink patch"),
        "remove_earrings": ("natural bare earlobes and ears, no earrings or jewelry, preserve the face and background", "earrings, ear jewelry, gold jewelry"),
    }
    seeds = range(310, 316)
    all_images = {}
    masks = {}
    for seed in seeds:
        source_path = SOURCE_DIR / f"indian_woman_bindi1_earrings1_seed{seed}.png"
        source = Image.open(source_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        all_images[(seed, "source")] = source
        center = bindi_center(source)
        for idx, (name, (prompt, negative)) in enumerate(conditions.items()):
            mask = soft_mask("bindi" if name == "remove_bindi" else "earrings", center=center)
            generated = pipe(prompt=prompt, negative_prompt=negative, image=source,
                              mask_image=mask, strength=1.0, num_inference_steps=40,
                              guidance_scale=4.5,
                              generator=torch.Generator("cuda").manual_seed(9100 + seed*2 + idx),
                              height=512, width=512).images[0]
            # Enforce the intended edit region: outside it retain the exact source pixels.
            output = Image.composite(generated, source, mask)
            all_images[(seed, name)] = output
            masks[(seed, name)] = mask
            output.save(OUT / f"{name}_seed{seed}.png")
            print(f"completed seed {seed}: {name}", flush=True)

    labels = ["source", "remove bindi", "remove earrings"]
    sheet = Image.new("RGB", (len(labels) * 512, len(seeds) * 530 + 25), "white")
    draw = ImageDraw.Draw(sheet)
    for i, label in enumerate(labels):
        draw.text((i * 512 + 4, 5), label, fill="black")
    for row, seed in enumerate(seeds):
        for col, name in enumerate(("source", "remove_bindi", "remove_earrings")):
            x, y = col * 512, 24 + row * 530
            sheet.paste(all_images[(seed, name)], (x, y))
            draw.text((x + 4, y + 512 + 2), f"seed {seed}", fill="black")
    sheet.save(OUT / "inpaint_pilot_contact_sheet.jpg", quality=94)
    (OUT / "settings.json").write_text(json.dumps({
        "model": MODEL, "source_dir": str(SOURCE_DIR), "same_source_per_pair": True,
        "seeds": [min(seeds), max(seeds)],
        "mask_edit_regions": {"bindi": "98px diameter ellipse centered on detected red mark", "earrings": [[74, 285, 169, 417], [361, 285, 455, 417]]},
        "steps": 40, "guidance_scale": 4.5,
        "compositing": "Generated pixels were pasted only inside the feathered target mask; original source pixels retained outside mask.",
        "edit_stage_only": "This checkpoint is used for creating image controls; it is not the face SAE's diffusion model.",
        "contact_sheet": str(OUT / "inpaint_pilot_contact_sheet.jpg"),
    }, indent=2) + "\n")
    print(f"wrote {OUT / 'inpaint_pilot_contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    main()
