# SDXL Base flower SAE intervention

This is the planned SDXL Base counterpart to the existing [SDXL Turbo flower experiment](../../flower_sae_intervention/README.md). It is a clean experiment scaffold; no Base images, activations, or results have been generated yet.

## Plan

1. Generate the same prompt and seed set with `stabilityai/stable-diffusion-xl-base-1.0` using a documented multi-step sampler configuration.
2. Capture the same U-Net block used by the Turbo experiment (`unet.down_blocks.2.attentions.1`) and save spatial activations.
3. Train a fresh SAE on Base activations. The Turbo SAE checkpoint is model-specific and should not be reused.
4. Rank and localize candidate flower features against the same C2C box targets, then inspect a sample of boxes.
5. Compare feature ablation with a no-op and activity-matched random-feature control. Report flower visibility and person/image preservation separately from pixel MAE.

## Comparison controls

- Keep prompt text, seeds, image dimensions, C2C labels, and train/validation/test assignments matched to the Turbo run.
- Report the Base checkpoint revision, sampler, scheduler, inference steps, guidance scale, dtype, and the exact hook location.
- Do not compare Base activations to Turbo SAE features. Fit and evaluate the Base SAE independently.
- Treat C2C boxes as working labels, and report any manual box audit separately.

## Artifacts

Store generated images, activation caches, checkpoints, metrics, and visualizations in this folder's ignored run directories. Commit source code, settings, concise reports, and selected comparison images only; do not commit model weights or large activation caches.
