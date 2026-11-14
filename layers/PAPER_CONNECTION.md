# Relation to Tinaz et al., arXiv:2504.15473

[The paper](https://arxiv.org/html/2504.15473v2) studies multi-step SD v1.4, with
SAEs tied to layers and timesteps. Its concept labeling uses segmentation-mask
overlap; its temporal intervention results cannot be identified with a U-Net's
down/mid/up paths. See sections 3.2, 4.1, and 4.3.

## What this run tests

`turbo_layer_depth_001` measures **network depth at a single denoising step**.
The four layer-specific pretrained dictionaries encode the same 1,200 Turbo
samples. Discovery selects a feature by same-seed, same-gender prompt-group
contrast; held-out seeds evaluate that frozen choice.

The follow-up visuals use the exact 48 feature selections from
`results/layerwise_group_effects.csv`. They display JET maps and signed
multipliers `-5 -3 -2 -1 0 1 2 3 5` on fixed evaluation seeds. These test a
feature's image effects at its own layer, not an early/middle/late progression.
Feature IDs refer to independent dictionaries. Prompt-group association and
colorful spatial overlap are not, by themselves, semantic or fairness labels.

## A connected follow-up design

1. Define concrete visual outcomes: hair, head covering, shirt, or a localized
   artifact. Use reviewed segmentation masks; keep race/gender as prompt
   conditions rather than inferring identities from generated faces.
2. For each layer-specific feature, compare an explicitly thresholded spatial
   activation map with the relevant mask. Choose all thresholds and candidates
   on discovery data, then evaluate localization on held-out seeds. Report small
   region size and mask quality, since face-box means can hide local cues.
3. Test both whole-map and mask-restricted edits. Record intended attribute
   changes and changes outside the target mask. Include no-op, random-feature,
   and image-quality controls.
4. To test denoising time, use a suitable multi-step model and record actual
   scheduler timesteps separately from U-Net block names. Existing
   `capture_trajectories.py` is a starting point for collection, not an executed
   temporal SAE experiment.
5. Validate SAE reconstruction and activation distributions at each time.
   Train matched stage-specific or pooled-stage SAEs where necessary; do not
   assume the current Turbo checkpoints apply to another model or timestep.

This would connect the current depth audit to a layer-by-time study of specific
visual attributes. It would still require outcome evaluation before claiming
that an intervention reduces bias.
