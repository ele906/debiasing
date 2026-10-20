# Eyes-only SAE fine-tuning

This is a separate 20-epoch experiment initialized from the original pretrained
SAE. Training updates the existing SAE weights; it does not add latent features
or train the image generator. Only residual vectors whose 16×16 cell centers
fall inside an accepted eye box contribute to the training loss. Race/gender
groups are sampled uniformly during each epoch.

## Held-out results

The selected checkpoint is epoch 20, chosen by macro-average validation eye
reconstruction MSE. The test comparison was not used for checkpoint selection.

| Test set | Images | Metric | Original SAE | Eye-fine-tuned SAE |
|---|---:|---|---:|---:|
| Eye-boxed | 90 | Eye-region MSE | 2.1007 | 1.7608 |
| Eye-boxed | 90 | Whole-face MSE | 2.1485 | 1.9282 |
| Eye-boxed | 90 | All-cell MSE | 2.1011 | 1.9750 |
| All face-boxed test images | 240 | Whole-face MSE | 2.1881 | 1.9671 |
| All face-boxed test images | 240 | Outside-face MSE | 2.1331 | 2.0567 |
| All face-boxed test images | 240 | All-cell MSE | 2.1491 | 2.0307 |

On the 240-image test set, eye-region MSE is computed only over the 90 images
with usable eye annotations. The other whole-face and all-cell metrics use all
240 images. The annotation boxes are automatic and unreviewed, so these results
are exploratory reconstruction measurements rather than proof that a latent
feature represents “eyes.”

## Files

- `heldout_metrics.csv` — compact original-versus-fine-tuned metrics.
- `heldout_reconstruction.json` — pooled and category-level held-out metrics.
- `eye_localized_features.csv` — features ranked by validation activation
  contrast inside eyes versus the rest of the face.
- `region_heatmaps/eyes/` — epoch-by-epoch activation overlays for the top
  three eye-ranked features.
- `training_history.json` — validation measurements at saved snapshots
  (epochs 0, 4, 8, 12, 16, 20). The per-epoch training losses were printed
  during the run but were not persisted after the run completed.
