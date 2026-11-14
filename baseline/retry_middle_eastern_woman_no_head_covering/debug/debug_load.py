import os, sys, time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("starting")
os.chdir('/n/fs/goose/ReNO')
sys.path.append('/n/fs/goose/sdxl-unbox')

log("importing torch")
import torch
log(f"torch imported, cuda available={torch.cuda.is_available()}")

log("importing diffusers")
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
log("diffusers imported")

log("importing SDLens")
from SDLens import HookedStableDiffusionXLPipeline
log("SDLens imported")

log("importing SAE")
from SAE import SparseAutoencoder
log("SAE imported")

CACHE_DIR = 'hf_cache'
DTYPE = torch.float16
DEVICE = torch.device('cuda')

log("loading VAE")
hooked_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=DTYPE, cache_dir=CACHE_DIR,
)
log("VAE loaded")

log("loading pipeline")
hooked_pipe = HookedStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", vae=hooked_vae, torch_dtype=DTYPE, variant="fp16",
    use_safetensors=True, cache_dir=CACHE_DIR,
)
log("pipeline loaded")

hooked_pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    hooked_pipe.pipe.scheduler.config, timestep_spacing="trailing",
)
log("moving pipeline to device")
hooked_pipe.pipe = hooked_pipe.pipe.to(DEVICE, DTYPE)
log("pipeline on device")

log("running a tiny generation")
generator = torch.Generator("cuda").manual_seed(0)
latents = torch.randn((1, 4, 64, 64), device=DEVICE, dtype=DTYPE)
with torch.no_grad():
    images, cache = hooked_pipe.run_with_cache(
        "a portrait of a person",
        latents=latents,
        generator=generator,
        num_inference_steps=4,
        guidance_scale=0.0,
        positions_to_cache=["unet.down_blocks.2.attentions.1"],
        save_input=True,
        save_output=True,
    )
log("generation done")
images.images[0].save("/n/fs/goose/baseline/retry_middle_eastern_woman_no_head_covering/debug/debug_test.png")
log("saved test image -- ALL GOOD")
