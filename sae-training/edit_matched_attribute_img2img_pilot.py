"""Try small paired bindi/earring edits while retaining one shared source portrait."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
from SDLens.hooked_sd_pipeline import HookedDiffusionAbstractPipeline
from SAE import SparseAutoencoder

MODEL = "stabilityai/sdxl-turbo"
VAE = "madebyollin/sdxl-vae-fp16-fix"
BLOCK = "unet.down_blocks.2.attentions.1"
SAE_DIR = ROOT / "sae-training/runs/face_sae_020/sae"
SOURCE = ROOT / "sae-training/runs/face_sae_020/bindi_earring_pilot_001/images/indian_woman_bindi1_earrings1_seed310.png"
OUT = ROOT / "sae-training/runs/face_sae_020/bindi_earring_img2img_pilot_001"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE).convert("RGB")
    conditions = {
        "keep_bindi_keep_earrings": ("same Indian woman, same face, same hair, same clothing, preserve the small red bindi and gold earrings", ""),
        "remove_bindi_keep_earrings": ("same Indian woman, same face, same hair, same clothing, smooth bare forehead with no bindi or forehead mark, preserve gold earrings", "bindi, red dot, forehead mark, forehead jewelry"),
        "keep_bindi_remove_earrings": ("same Indian woman, same face, same hair, same clothing, preserve the small red bindi, bare earlobes with no earrings", "earrings, ear jewelry"),
        "remove_bindi_remove_earrings": ("same Indian woman, same face, same hair, same clothing, smooth bare forehead with no bindi or forehead mark, bare earlobes with no earrings", "bindi, red dot, forehead mark, forehead jewelry, earrings, ear jewelry"),
    }
    settings = {"source": str(SOURCE), "model": MODEL, "vae": VAE, "block": BLOCK,
                "sae_checkpoint": str(SAE_DIR), "steps": 4, "guidance_scale": 1.5,
                "strength": 0.55, "same_source_image": True, "same_random_seed": 771,
                "interpretation": "pilot only; visually verify attribute changes and collateral changes before SAE inference"}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(VAE, **load)
    raw_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL, vae=vae, variant="fp16", **load)
    raw_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(raw_pipe.scheduler.config, timestep_spacing="trailing")
    pipe = HookedDiffusionAbstractPipeline(raw_pipe)
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(SAE_DIR)).float().cuda().eval()
    sae.requires_grad_(False)
    panels, activations = [], {}
    for name, (prompt, negative) in conditions.items():
        kwargs = {"prompt": prompt, "negative_prompt": negative or None, "image": source,
                  "strength": settings["strength"], "num_inference_steps": settings["steps"],
                  "guidance_scale": settings["guidance_scale"],
                  "generator": torch.Generator("cuda").manual_seed(settings["same_random_seed"]),
                  "height": 512, "width": 512}
        with torch.no_grad():
            result, cache = pipe.run_with_cache(**kwargs, positions_to_cache=[BLOCK], save_input=True)
            residual = (cache["output"][BLOCK] - cache["input"][BLOCK])[:, -1].permute(0, 2, 3, 1).float()
            code = sae.encode(residual)[0].float().cpu().numpy()
        image = result.images[0]
        image.save(OUT / f"{name}.png")
        activations[name] = code
        panels.append((name, image))
        print(f"completed {name}", flush=True)
    np.savez_compressed(OUT / "face_sae_activations.npz", **{k: v for k, v in activations.items()})
    sheet = Image.new("RGB", (1024, 1060), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (name, image) in enumerate(panels):
        x, y = (idx % 2) * 512, (idx // 2) * 530
        draw.text((x + 4, y + 3), name.replace("_", " "), fill="black")
        sheet.paste(image.resize((512, 512), Image.Resampling.LANCZOS), (x, y + 22))
    sheet.save(OUT / "img2img_pilot_contact_sheet.jpg", quality=93)
    print(f"wrote {OUT / 'img2img_pilot_contact_sheet.jpg'}", flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires a CUDA GPU")
    main()
