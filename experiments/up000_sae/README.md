# Independent SDXL Turbo SAE study at up.0.0

See [AUDIT.md](AUDIT.md) for the checkpoint, hook and matched mid.0 design. [jobs.txt](jobs.txt) records SLURM submissions.

## Execution

- `train.sbatch` captures the matched 1,200 residual grids and calls the existing face-region trainer. Capture resumes from verified residual files. An interrupted fit is archived and retrained from the pretrained checkpoint because the historical trainer saves weights without optimizer or RNG state.
- `discovery.sbatch` checks the fine-tuned checkpoint hashes, ranks the full dictionary, selects the historical candidate-filtered top ten, and renders seed-185 activation overlays. That discovery seed range overlaps SAE training and validation. The 30–34 intervention seeds are separate.
- `sweep.sbatch pilot` uses one array task per selected feature. Each completed condition has a PNG, activation maps and hash-verified JSON record. Identical array submissions resume missing conditions. Validation requires a pixel-identical unhooked control, zero applied perturbation in no-op cases, active edits for every selected feature, and matching archived mid.0 controls.
- After a successful pilot, submit `sbatch --array=0-9%10 experiments/up000_sae/sweep.sbatch full`. The worker checks the validation result before generating any images. The full sweep uses the historical 29 multipliers.
- The evaluator compares MediaPipe detections and RGB MAE, including paired change inside and outside the convex hull of baseline face landmarks. It reports seed-level standard errors and contrasts within shared residual-relative L2 bins using matched prompt/seed pairs. The historical human review CSV remains the source for existing ratings. MediaPipe detections measure detector success. Human structural validity and semantic interpretation require review of the galleries.

## GitHub contents

The GitHub copy includes the report, experiment scripts, rankings, manifests, training and evaluation summaries, activation overlays, and selected edit galleries. Manifest paths to individual sweep images refer to the cluster workspace. The full image and mask collections, residual caches, and checkpoint weights remain on the cluster. Executing the SLURM scripts requires the existing cluster dependencies and historical mid.0 artifacts described in the audit.
