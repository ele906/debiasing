# Script index

This folder is the root-level map for the scripts used most often. The canonical scripts remain beside their experiments for now because they depend on local project layouts and import paths. Use these categories to find and run them without copying or moving working code.

## Start generation

- [`start/README.md`](start/README.md): simple prompt generation and the existing SDXL Turbo SAE generation/activation-capture runner.
- [`generate/README.md`](generate/README.md): interactive notebooks for prompt batches and a Turbo/Base visual comparison.

## Train an SAE

- [`sae-train/README.md`](sae-train/README.md): C2C box-target preparation, residual capture, fine-tuning, and frozen-SAE baseline.

## Activation features

- [`activation-features/README.md`](activation-features/README.md): activation generation, feature analysis, and visualization scripts.

Use the documented commands from the repository root. The `sae-training/` directory is itself a separate Git repository; its commits belong to that repository, while root-level `scripts/` and `sdxl-base/` files belong to `debiasing`.
