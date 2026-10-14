"""Small helpers shared by the interactive SDXL generation notebooks."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler


MODELS = {
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "SDXL Base": "stabilityai/stable-diffusion-xl-base-1.0",
}


def model_defaults(model_name: str) -> dict:
    if model_name == "SDXL Turbo":
        return {"steps": 1, "guidance": 0.0, "size": 512}
    if model_name == "SDXL Base":
        return {"steps": 30, "guidance": 5.0, "size": 1024}
    raise ValueError(f"Unknown model: {model_name}")


def load_pipeline(model_name: str):
    """Load a model once. First use may download weights from Hugging Face."""
    if model_name not in MODELS:
        raise ValueError(f"Choose one of: {', '.join(MODELS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this notebook in its current setup.")
    dtype = torch.float16
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if model_name == "SDXL Turbo":
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(MODELS[model_name], **kwargs)
    if model_name == "SDXL Turbo":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_batch(
    pipe,
    model_name: str,
    prompt: str,
    count: int,
    seed: int,
    steps: int,
    guidance: float,
    output_root: str | Path = "outputs/interactive_generation",
    run_name: str = "batch",
):
    """Generate a reproducible batch and save images plus the run settings."""
    if not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if count < 1:
        raise ValueError("Image count must be at least one.")
    defaults = model_defaults(model_name)
    if model_name == "SDXL Turbo":
        # Turbo is distilled for very few steps; use its recommended one-step profile.
        steps, guidance = 1, 0.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "model_name": model_name,
        "model_id": MODELS[model_name],
        "prompt": prompt,
        "count": count,
        "first_seed": seed,
        "seeds": list(range(seed, seed + count)),
        "steps": steps,
        "guidance_scale": guidance,
        "width": defaults["size"],
        "height": defaults["size"],
        "dtype": "float16",
        "scheduler": pipe.scheduler.__class__.__name__,
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    images = []
    for i in range(count):
        print(f"Generating image {i + 1}/{count} (seed {seed + i})…", flush=True)
        generator = torch.Generator(device="cuda").manual_seed(seed + i)
        result = pipe(
            prompt=prompt,
            height=defaults["size"],
            width=defaults["size"],
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]
        images.append(image)
        image.save(run_dir / f"image_{i:02d}_seed_{seed + i}.png")
    return images, run_dir, settings
