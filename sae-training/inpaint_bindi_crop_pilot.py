"""Remove one bindi with crop-aware inpainting to preserve local detail."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLInpaintPipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SOURCE = ROOT / "sae-training/runs/face_sae_020/bindi_earring_pilot_001/images/indian_woman_bindi1_earrings1_seed310.png"
OUT = ROOT / "sae-training/runs/face_sae_020/inpaint_bindi_crop_pilot_001"


def find_mark_center(image):
    arr = np.asarray(image.convert("RGB"))
    crop = arr[140:260, 170:350]
    is_red = ((crop[..., 0] > 100) & (crop[..., 0] > crop[..., 1] * 1.4) &
              (crop[..., 0] > crop[..., 2] * 1.3) & (crop[..., 1] < 150))
    y, x = np.where(is_red)
    if not len(x):
        raise RuntimeError("No red mark found in expected forehead region")
    return int(np.median(x) + 170), int(np.median(y) + 140)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    cx, cy = find_mark_center(source)
    big = Image.new("L", (2048, 2048), 0)
    ImageDraw.Draw(big).ellipse(((cx-28)*4, (cy-28)*4, (cx+28)*4, (cy+28)*4), fill=255)
    mask = big.resize((512, 512), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(2))
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"),
                local_files_only=True, use_safetensors=True, variant="fp16")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL, **load)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        edited = pipe(prompt="photorealistic continuous natural forehead skin, same person, no forehead ornament",
                      negative_prompt="bindi, red dot, forehead mark, jewelry, colored patch, artifact",
                      image=source, mask_image=mask, padding_mask_crop=96,
                      strength=1.0, num_inference_steps=35, guidance_scale=4.5,
                      generator=torch.Generator("cuda").manual_seed(9510),
                      height=512, width=512).images[0]
    edited.save(OUT / "remove_bindi.png")
    source.save(OUT / "source.png")
    sheet = Image.new("RGB", (1024, 540), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), "source", fill="black")
    draw.text((516, 4), "crop-aware inpaint", fill="black")
    sheet.paste(source, (0, 22))
    sheet.paste(edited, (512, 22))
    sheet.save(OUT / "contact_sheet.jpg", quality=94)
    (OUT / "settings.json").write_text(json.dumps({
        "model": MODEL, "source": str(SOURCE), "feature_model": "not used for editing",
        "mask_center": [cx, cy], "mask_diameter": 56, "padding_mask_crop": 96,
        "steps": 35, "guidance_scale": 4.5,
    }, indent=2) + "\n")
    print(str(OUT / "contact_sheet.jpg"), flush=True)


if __name__ == "__main__":
    main()
