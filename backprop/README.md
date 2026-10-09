# Simple-prompt images and manual backprop

Requested dataset: **6 racial prompt categories × 2 genders × 100 images = 1,200 images**.
The identical prompt template for all groups is `a picture of a {race} {gender}`.
Categories: black, white, east asian, middle eastern, indian, hispanic; man/woman.
Categories are labels in the generation prompts, not identities inferred from pixels.

Existing `../baseline/images` contains 600 portraits (50 per cell), but its prompt
adds expression, no facial coverings, a white shirt, and a white background.
Those images are a separate engineered-prompt comparison dataset; they are not
substituted for this simple-prompt baseline. Earlier ReNO outputs are occupation
or multi-person optimization runs, not this matched 12-cell baseline.

## Generation (submitted to Slurm)

Uses cached SDXL Turbo, the fp16 VAE, one inference step, guidance zero, Euler
ancestral scheduler with trailing timesteps, 512×512 images, and the existing
pretrained down.2.1 k10 SAE. No models are downloaded or trained.

```bash
sbatch /n/fs/goose/backprop/generate.slurm
squeue -u el8403
```

The job resumes missing image/feature pairs. It refuses to mix differing settings.
Logs are in `logs/`; `generation_summary.json` is written only when complete.
Outputs:

- `images/<race>_<gender>/seed000.png` through `seed099.png`.
- `contact_sheets/`: one labelled grid of all 100 images per group.
- `manifest.csv`: prompt, seed, split, and file paths for every sample.
- `generation_settings.json`: model/software versions and SAE weight hash.
- `features/`: pretrained SAE spatial mean/max/active-fraction vectors per image.
- `sae_features.npz`: all pooled feature matrices, `[1200, 5120]`, in manifest order.
- `annotations.csv`: blank human annotation sheet. Never overwritten on resume.

Generation collects the residual of `unet.down_blocks.2.attentions.1`, encodes
it with the pretrained SAE, and pools the features. It does not train a new SAE.
Seeds are shared across all groups. The same seeds belong to train (60), validation
(20), or test (20) across all 12 groups, avoiding seed leakage between splits.

## Stage 2: explore the pretrained SAE features

The GPU scripts require the existing [sdxl-unbox](https://github.com/surkovv/sdxl-unbox)
checkout at `../sdxl-unbox`, its pretrained SAE checkpoint, and cached SDXL Turbo
model weights. The upstream checkout is a separate Git repository, not bundled
in this commit. The analysis script only needs NumPy and the locally generated
`sae_features.npz`. Large datasets, weights, and heatmap run outputs stay local.

`analyze_sae_features.py` adapts the ranking method from
`../sdxl-unbox/scripts/sae_bias_experiment.py` to this dataset. It uses spatial
mean activations by default, ranks positive within-gender race contrasts on
discovery seeds, checks their effects on validation seeds, and reserves test seeds.
Use `--pooling max --output backprop/sae_analysis_max` for the old classifier's
spatial-max representation. Feature IDs retain the original pretrained SAE meaning.

The initial analysis is in [sae_analysis/REPORT.md](sae_analysis/REPORT.md).
Full top-20 lists and spatial/image activity measurements are in its CSV files.
These are associations with prompt groups, not covering-specific feature labels.

`sae_heatmaps.py` adapts the visual workflow from
`../baseline/sae_top10_heatmaps.ipynb`. It replays three discovery seeds per cell,
checks exact image reproduction, and overlays the three top-ranked feature maps.
It saves raw maps and activation measurements so overlay brightness is not used
to compare magnitudes. It imports the existing SAE implementation, keeping the
pretrained checkpoint and feature IDs intact.

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python backprop/analyze_sae_features.py \
  --output backprop/sae_analysis_new
sbatch backprop/sae_heatmaps.slurm
```

The corrected heatmap job is **4074519**, with log
`logs/backprop_sae_maps-4074519.out`. It was queued awaiting a GPU when Stage 2
was published; the previous job failed on an import-path issue that is now fixed.
Check `squeue -j 4074519` for current status. When complete, inspect
[sae_heatmaps/REPORT.md](sae_heatmaps/REPORT.md). Analysis and visualization do not
train an SAE or classifier and do not suppress any features.

## Backprop code (written, not trained)

`numpy_backprop.py` adapts the class approach: model dictionary, forward pass,
reverse manual gradients, and SGD updates. Inputs are `[features, batch]`.
It supports logistic regression or a one-hidden-layer ReLU classifier. Stable
softmax cross-entropy operates on logits; there is no separate softmax layer.
Input gradients are returned so the attribute probe can suggest feature candidates.
Weight decay applies to weights only. Checkpoints contain numeric arrays, not
pickled function objects.

`check_gradients.py` verifies analytical parameter/input gradients with numerical
finite differences on tiny synthetic arrays. It performs no training.

## Later: label and train an attribute probe

Review `contact_sheets/` and full-resolution images. In `annotations.csv`, set
`head_covering` to `1` (present), `0` (absent), or `uncertain`. Blank/uncertain
examples are excluded. Define consistently whether hats, turbans, scarves, and
hoods count; presence itself is not a harm or racism label. Record artifacts and
notes. Label covered/uncovered examples within groups to distinguish covering
features from prompt-group features. Both train and validation need both classes.

Nothing automatically assigns race from pixels or treats racial-category
classification accuracy as reduced bias.

**Run this only when ready to train:**

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python backprop/train_probe.py
```

The default probe is linear and trains on SAE spatial means. Optional
`--hidden-size 32` adds a hidden layer; `--pooling max` uses spatial maxima.
Normalization uses training samples only. Best checkpoint selection uses
validation loss; test metrics are computed afterward. A fresh output directory
is required. It saves model, training history, metrics, and candidate feature IDs.
Candidates combine within-race/gender covering contrasts and manual input-gradient
contributions on training data. These are associations/local attributions, not
verified semantic labels or causal effects. Candidate selection may be impossible
if covering presence is perfectly confounded with prompt group; collect targeted
control data rather than interpreting group separation as a covering feature.

## Later: fixed suppression, independent of classifier training

Choose feature IDs after inspecting training/validation associations and controls.
For example, once you have evidence for feature IDs, replace the illustrative IDs:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python backprop/fixed_suppression.py \
  --features 138 527 --output backprop/suppression_exploratory
```

**138 and 527 are illustrative historical IDs, not validated covering controls.**
The script edits each feature separately on three held-out seeds per cell, with
strengths 0, 0.5, 1, and 2. It uses the original prompts and independent seeded
noise/scheduler streams. Strength 1 must match the fresh replay exactly; baseline
replay differences are reported. The pretrained SAE and diffusion weights stay fixed.

The edit is `block_output + (strength - 1) * feature_activation * decoder_direction`.
It preserves the SAE reconstruction error; inactive features stay inactive.
Comparison grids, numerical image-change metrics, and blank human review sheets
are saved. Pixel changes or lower probe scores are not evidence of improvement;
review covering presence, prompt fidelity, hair/facial damage, and artifacts.

Fixed suppression itself does not need backprop. Backprop trains the probe that
suggests candidate features; the diffusion hook performs the actual image edit.
Learning suppression strengths through the generated image would require a
separate differentiable generator and validated attribute loss, and is not
implemented or claimed by this fixed-intervention experiment.
