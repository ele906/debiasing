"""
Generate init->final image trajectories for a small set of prompts across many
seeds, for later racial-composition analysis (see racial_composition_analyze.py).

For each prompt, runs `n_seeds` independent ReNO optimizations of
`n_iters` iterations each, saving the decoded image at every iteration
(via LatentNoiseTrainer's save_all_images option). Output layout:

    {save_dir}/{prompt_slug}/seed{N}/{iteration}.png

Usage:
    python scripts/racial_composition_generate.py --model sdxl-turbo \\
        --prompts "A photo of a CEO" "A photo of a nurse" "A photo of a software engineer" \\
        --n_seeds 20 --save_dir outputs/racial_composition
"""
import argparse
import logging
import os
import re

import torch
from pytorch_lightning import seed_everything

from arguments import parse_args
from models import get_latent_shape, get_model
from rewards import get_reward_losses
from training import BiasCorrector, LatentNoiseTrainer, get_optimizer


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main():
    base_args = parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "A photo of a CEO",
            "A photo of a nurse",
            "A photo of a software engineer",
        ],
    )
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument(
        "--save_dir", type=str, default="outputs/racial_composition"
    )
    sweep_args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

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

    bias_corrector = None
    if base_args.enable_debias:
        bias_corrector = BiasCorrector(
            base_args.debias_direction_path, base_args.debias_scale, device, dtype
        )
        logging.info(
            f"debias correction ON: direction={base_args.debias_direction_path} "
            f"scale={base_args.debias_scale}"
        )
    else:
        logging.info("debias correction OFF (baseline run)")

    for prompt in sweep_args.prompts:
        prompt_dir = os.path.join(sweep_args.save_dir, slugify(prompt))
        for seed in range(sweep_args.n_seeds):
            save_dir = os.path.join(prompt_dir, f"seed{seed}")
            done_marker = os.path.join(save_dir, "best_image.png")
            if os.path.exists(done_marker):
                logging.info(f"skipping prompt={prompt!r} seed={seed} (already done: {save_dir})")
                continue
            seed_everything(seed)
            os.makedirs(save_dir, exist_ok=True)

            trainer = LatentNoiseTrainer(
                reward_losses=reward_losses,
                model=pipe,
                n_iters=base_args.n_iters,
                n_inference_steps=base_args.n_inference_steps,
                seed=seed,
                save_all_images=True,
                device=device,
                regularize=base_args.enable_reg,
                regularization_weight=base_args.reg_weight,
                grad_clip=base_args.grad_clip,
                log_metrics=True,
                bias_corrector=bias_corrector,
            )

            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents.clone(), requires_grad=True)
            optimizer = get_optimizer(
                base_args.optim, latents, base_args.lr, base_args.nesterov
            )

            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, save_dir, multi_apply_fn=None
            )
            init_image.save(os.path.join(save_dir, "init_image.png"))
            best_image.save(os.path.join(save_dir, "best_image.png"))
            logging.info(
                f"prompt={prompt!r} seed={seed} done -> {save_dir} "
                f"(init={init_rewards}, best={best_rewards})"
            )


if __name__ == "__main__":
    main()
