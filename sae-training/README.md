# Concept-supervised SAE experiment

## Face SAE reconstruction images

`fine_tune_face_sae.py` now defaults to 20 epochs and saves SAE snapshots at epochs 0, 4, 8, 12, 16, and 20. Epoch 0 is the pretrained SAE before fine-tuning. The snapshots let the SDXL pipeline render the same held-out prompts and seeds through each checkpoint.

Run fine-tuning into a new run directory, then render one held-out test portrait from each of the 12 prompt groups:

```bash
python sae-training/fine_tune_face_sae.py --epochs 20 --output sae-training/runs/face_sae_020
python sae-training/render_epoch_reconstructions.py sae-training/runs/face_sae_020
```

Each sample gets a comparison sheet under `reconstructions/<sample_id>/comparison.jpg`; `reconstructions/all_samples.jpg` gives an overview. Rendering uses the saved prompt and seed for every checkpoint. The saved source image is shown as `original`; epoch 0 and later columns are generated with the corresponding SAE reconstruction hook. This is an SDXL image comparison of SAE-mediated generation, rather than a direct VAE decode of a residual tensor.

## Goal

Test whether reviewed concept labels help an SAE represent *kimono* and *flower* more directly than the current reconstruction-trained SAE, while preserving useful reconstruction quality. This is a separate experiment from attenuating one feature in the frozen SAE.

## What exists now

- `backprop/sae_features.npz` contains pooled activations for 1,200 SDXL portraits from the existing 5,120-feature Top-K SAE (k=10), at `unet.down_blocks.2.attentions.1`.
- The current SAE remains frozen in all completed concept work. Its feature 5046 for kimono and 2230 for flower came from Florence-2 labels, not human review.
- The detector labels in `concept/concept_predictions.csv` are weak supervision: false negatives and hallucinations are possible. The existing kimono validation comparison has no matched within-prompt validation group; flower has only one. The ablation pilot used only four same-prompt images per concept (kimono: 2 positive/2 negative) and did not show reliable object-specific removal.
- `backprop/annotations.csv` has blank human review fields. `annotations_template.csv` is a separate working copy with one row per source image and blank `kimono_present` / `flower_present` labels. It does not promote detector predictions to ground truth.

## Experiment stages

1. **Review labels.** Inspect images and fill `kimono_present` and `flower_present` with `1`, `0`, or `uncertain`. Blank and uncertain are excluded. Use consistent visual definitions, record ambiguous cases, and label both concepts across prompt groups. Keep detector predictions as optional suggestions in a separate column/source.
2. **Check feasibility before training.** Report class counts by split and race/gender prompt cell. Require train and validation positives and negatives; require both classes within at least two prompt cells for a first controlled result. If labels are too sparse or correlated with prompt group, gather balanced targeted examples instead of fitting on detector outputs.
3. **Run a frozen-SAE probe baseline first.** Train the existing linear probe (`backprop/train_probe.py`) on reviewed pooled SAE activations for each label. This measures whether current SAE codes already contain linearly accessible concept information and yields candidate features. It does not train an SAE and its candidate IDs are associations, not causal controls.
4. **Train the concept-guided SAE variant.** Start from the same residual activations/checkpoint and split. Optimize reconstruction loss plus Top-K sparsity and an auxiliary concept prediction loss on the SAE codes. Keep the decoder/reconstruction path active for every example; apply concept loss only to reviewed labels. Compare against the frozen baseline at matched width, k, optimizer steps, and data. Tune auxiliary-loss weight on validation only.
5. **Evaluate without claiming object control.** Report reconstruction MSE / explained variance, sparsity and dead-feature rate, probe AUROC and average precision by concept, within-prompt-cell metrics, and performance on held-out test seeds. Inspect top activating images and spatial maps. Only then test selected features with same-prompt controls, matched random features, and blinded image review.

## Guardrails

- Split by the existing sample split; never use test labels for model choice, feature ranking, or auxiliary-loss tuning.
- Fit no concept loss on blank, uncertain, or detector-only labels unless a separate weak-label experiment is explicitly named and reported as such.
- Report pooled and within-prompt-group performance. A global score can learn the race/gender prompt cell instead of the object.
- Compare the new SAE to the frozen checkpoint and a probe on frozen SAE activations. A better probe score alone does not establish better feature disentanglement or causal image editing.
- Keep checkpoint, settings, label version/hash, and outputs in a unique run directory under this folder. Do not overwrite prior runs or copy model weights/datasets into Git.

## Current readiness

**Not ready for concept-supervised training.** The needed human concept labels are not present. The detector outputs are useful for prioritizing image review, but the small and confounded validation groups are inadequate as targets for claiming a concept-specialized SAE. First complete and audit `annotations_template.csv`; then implement the auxiliary-loss trainer once label counts support the split and group checks above.

The existing non-training intervention pilot is documented in [tuning/README.md](../tuning/README.md), and the broader detector/SAE evidence is in [concept/README.md](../concept/README.md).
