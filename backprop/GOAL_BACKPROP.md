# Backprop toward a concept-removal goal

The current backprop code trains a small probe on pooled SAE activations. Its gradients identify which SAE activations influence a human-reviewed concept score. The existing diffusion intervention then edits a fixed feature contribution. It does **not** backpropagate through the generated image or prove that the object is removed.

The SAE reconstruction/sparsity loss is for learning a useful sparse representation; it does not express “remove the flower/turban/kimono.” The object goal needs reviewed presence labels and its own loss.

The next goal is concept-specific: lower the score for one visible object while preserving the rest of the portrait. For a reviewed image and an SAE activation vector `z`, a first probe objective is:

```text
L_probe = binary_cross_entropy(probe(z), object_present)
```

Backpropagation through the probe gives `dL_probe/dz`, which can rank candidate SAE features. Then test those features in diffusion with fixed strength sweeps and random-feature controls. Evaluate the object itself and collateral changes by human review. A low probe score or changed pixels alone does not establish removal.

An end-to-end generator objective would need a differentiable concept score on the edited output plus preservation terms, for example:

```text
L = BCE(probe(SAE(h_edited)), absent) + lambda_preserve * L_preserve + lambda_edit * L_intervention_size
```

Here `h_edited` is the intervened UNet activation. `L_preserve` should penalize damage outside the object region and changes to the face or prompt; `L_intervention_size` discourages unnecessarily large feature edits.

Optimizing feature scales against that loss through SDXL is a later step. It is not implemented yet: the current pipeline hook and one-step outputs have not been verified for stable gradients through the full generator. The objective also needs human-reviewed concept labels and a validated differentiable object score before optimizing anything.

## First annotation batch

[The 60-image review packet](sae_review_60/README.md) samples five images from each of the 12 race/gender prompt groups: four training images and one validation image per group. It asks reviewers to label visible turban, flower object/accessory, and kimono presence. The labels are blank until a person reviews the contact sheets. This is a small pilot for checking the workflow; rare concepts may have too few positive examples to train or validate a useful probe.

After labels are reviewed, `train_probe.py` accepts the target column:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python backprop/train_probe.py \
  --annotations backprop/sae_review_60/annotations.csv \
  --label-column flower_present \
  --output backprop/probe_flower_60
```

The command will refuse to train if either the training or validation split lacks both classes. With only 60 images, that may be the correct outcome; gather more reviewed examples if so. The current packet preserves the original held-out test split by sampling only train and validation images.
