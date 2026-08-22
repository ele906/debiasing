"""
Recreates the plain (non-fairness) ReNO single-prompt trajectory demo that
produced outputs/single/sdxl-turbo_0_.../goose/{0..49,init,best}.png, for a
new prompt. main.py (the original task-dispatch entrypoint referenced in
README.md) isn't present in this repo, so this reimplements the "single"
task path directly from the same components racial_composition_generate.py
uses, with the same default args test.ipynb used for the goose demo.

Usage:
    python scripts/generate_single_demo.py --prompt parrot
"""
import argparse
import os

import torch
from pytorch_lightning import seed_everything

from arguments import parse_args
from models import get_latent_shape, get_model
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer


def main():
    base_args = parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt", type=str, required=True)
    sweep_args, _ = parser.parse_known_args()

    device = torch.device("cuda")
    dtype = torch.float16 if base_args.dtype == "float16" else torch.float32

    reward_losses = get_reward_losses(base_args, dtype, device, base_args.cache_dir)
    pipe = get_model(
        base_args.model,
        dtype,
        device,
        base_args.cache_dir,
        base_args.memsave,
        base_args.cpu_offloading,
    )
    shape = get_latent_shape(pipe, base_args.model)

    config_str = (
        f"{base_args.model}_{base_args.seed}_lr{base_args.lr}_gc{base_args.grad_clip}"
        f"_iter{base_args.n_iters}_reg{base_args.reg_weight}"
        f"_pickscore{base_args.pickscore_weighting}_clip{base_args.clip_weighting}"
        f"_hps{base_args.hps_weighting}_imagereward{base_args.imagereward_weighting}"
    )
    save_dir = os.path.join(base_args.save_dir, "single", config_str, sweep_args.prompt)
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(base_args.seed)
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=base_args.n_iters,
        n_inference_steps=base_args.n_inference_steps,
        seed=base_args.seed,
        save_all_images=True,
        device=device,
        regularize=base_args.enable_reg,
        regularization_weight=base_args.reg_weight,
        grad_clip=base_args.grad_clip,
        log_metrics=True,
    )

    init_latents = torch.randn(shape, device=device, dtype=dtype)
    latents = torch.nn.Parameter(init_latents.clone(), requires_grad=True)
    optimizer = get_optimizer(base_args.optim, latents, base_args.lr, base_args.nesterov)

    init_image, best_image, init_rewards, best_rewards = trainer.train(
        latents, sweep_args.prompt, optimizer, save_dir, multi_apply_fn=None
    )
    init_image.save(os.path.join(save_dir, "init_image.png"))
    best_image.save(os.path.join(save_dir, "best_image.png"))
    print(f"prompt={sweep_args.prompt!r} done -> {save_dir} (init={init_rewards}, best={best_rewards})")


if __name__ == "__main__":
    main()
