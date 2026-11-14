# SAE training

The canonical training code currently lives in the separate `sae-training/` Git repository. Run from the `debiasing` repository root:

```bash
python sae-training/prepare_box_targets.py
sbatch sae-training/capture_residuals.slurm
# After residual capture completes:
sbatch sae-training/train_box_sae.slurm
sbatch sae-training/train_box_baseline.slurm
```

The scripts currently use the SDXL Turbo activation setup described in [`sae-training/README.md`](../../sae-training/README.md). The planned SDXL Base experiment needs its own residuals and a newly trained SAE; do not load the Turbo checkpoint for Base activations.
