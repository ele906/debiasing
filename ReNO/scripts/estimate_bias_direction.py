"""
Estimate the direction in noise/latent space that ReNO's reward optimization
systematically drifts toward, for use by BiasCorrector (training/bias_correction.py).

For a list of prompts (default: assets/bias_probe_prompts.txt, occupation-style
prompts where representational bias tends to show up) and a range of seeds, this
runs ordinary ReNO optimization and records how far the best latents moved from
their random initialization. The *mean* drift across all (prompt, seed) samples
is the "bias direction": the component of the drift that's consistent across
prompts, rather than idiosyncratic to any single one.

This only tells you that the drift is systematic, not that it is demographic
bias specifically -- validate what it actually shifts (e.g. with a demographic
classifier on generated images, with vs. without correction) before trusting it.

Usage:
    python scripts/estimate_bias_direction.py --model sdxl-turbo --n_seeds 8 \\
        --output bias_directions/sdxl-turbo.pt
"""
import argparse
import logging
import os

import torch
from pytorch_lightning import seed_everything

from arguments import parse_args
from models import get_latent_shape, get_model
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer


def main():
    # Reuse the main arg parser for model/reward settings, then layer
    # estimation-specific args on top.
    base_args = parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--prompts_file", type=str, default="assets/bias_probe_prompts.txt"
    )
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--output", type=str, default="bias_directions/default.pt")
    est_args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    device = torch.device("cuda")
    dtype = torch.float16 if base_args.dtype == "float16" else torch.float32

    with open(est_args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

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

    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=base_args.n_iters,
        n_inference_steps=base_args.n_inference_steps,
        seed=base_args.seed,
        device=device,
        regularize=base_args.enable_reg,
        regularization_weight=base_args.reg_weight,
        grad_clip=base_args.grad_clip,
        log_metrics=False,
    )

    drifts = []
    drift_norms = []
    for prompt in prompts:
        for seed in range(est_args.n_seeds):
            seed_everything(seed)
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents.clone(), requires_grad=True)
            optimizer = get_optimizer(
                base_args.optim, latents, base_args.lr, base_args.nesterov
            )
            trainer.train(latents, prompt, optimizer, save_dir=None, multi_apply_fn=None)
            drift = (trainer.best_latents.to(device) - init_latents).flatten()
            drifts.append(drift)
            drift_norms.append(drift.norm().item())
            logging.info(
                f"prompt={prompt!r} seed={seed} drift_norm={drift_norms[-1]:.3f}"
            )

    all_drifts = torch.stack(drifts)
    mean_drift = all_drifts.mean(dim=0)
    mean_sample_norm = sum(drift_norms) / len(drift_norms)
    consistency = mean_drift.norm().item() / mean_sample_norm
    logging.info(
        f"Mean drift norm: {mean_drift.norm().item():.3f}, "
        f"mean per-sample drift norm: {mean_sample_norm:.3f}, "
        f"consistency ratio: {consistency:.3f} "
        "(near 0 => no consistent direction across prompts/seeds -- BiasCorrector "
        "won't have much reliable signal to work with; closer to 1 => highly "
        "consistent direction)"
    )

    os.makedirs(os.path.dirname(est_args.output) or ".", exist_ok=True)
    torch.save(mean_drift.reshape(shape).cpu(), est_args.output)
    logging.info(f"Saved bias direction to {est_args.output}")


if __name__ == "__main__":
    main()
