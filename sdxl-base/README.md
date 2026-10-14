# SDXL Base experiments

This folder is the home for experiments that use the standard SDXL Base checkpoint. The existing `flower_sae_intervention/` results use SDXL Turbo and remain the comparison baseline; Turbo results should not be presented as Base results.

## First experiment: flower SAE intervention

See [`flower_sae_intervention/README.md`](flower_sae_intervention/README.md) for the proposed matched Base run. Keep its model settings, activation cache, SAE checkpoint, results, and images inside that experiment folder. Train a new SAE on Base activations rather than reusing the Turbo SAE.

The initial comparison should hold prompts, seeds, U-Net block, image size, and evaluation criteria constant where possible. Record the Base sampler, step count, guidance scale, scheduler, and checkpoint revision. The Base model's ordinary multi-step sampling is a separate condition from the earlier four-step Turbo replay.

Model: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). Check its model card and license before redistributing generated artifacts or weights.
