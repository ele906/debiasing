# What happened in the signed layerwise SAE sweep

## Short version

We took the **exact 48 layer/group feature selections** from
`layerwise_group_effects_jet.png` and tested each selected SAE feature with
signed multipliers:

```text
-5, -3, -2, -1, 0, 1, 2, 3, 5
```

This produced **1,296 edited images**:

```text
4 SAE layers × 12 prompt groups × 3 held-out seeds × 9 multipliers = 1,296
```

The images and JET overlays are available from [the visual index](README.md).
The main result is that we now have an exact, reproducible way to compare what
the selected feature does at every layer. This run does **not** show that a
feature has a human-semantic racial label, that an edit reduces bias, or that
an edit improves image quality.

## What was already known

The source plot (`../layerwise_group_effects_jet.png`) compared four
layer-specific pretrained SDXL Turbo SAEs:

| Short name | U-Net block |
|---|---|
| `down.2.1` | `unet.down_blocks.2.attentions.1` |
| `mid.0` | `unet.mid_block.attentions.0` |
| `up.0.0` | `unet.up_blocks.0.attentions.0` |
| `up.0.1` | `unet.up_blocks.0.attentions.1` |

For each layer and prompt group, the earlier analysis used discovery seeds
100–149 to choose the single feature with the largest same-seed,
same-gender face-region contrast against the other five race prompts. It then
reported that fixed choice on held-out seeds 150–199. There are 48 choices:
four layers times six race prompts times two gender prompts.

The same feature number in different rows or layers does not mean the same
thing. Each SAE has a separate dictionary.

## What this run did

This follow-up did not choose a new feature from the held-out examples. It read
the frozen choices from `../layerwise_group_effects.csv`, then replayed three
held-out seeds, 150, 151, and 152, for every prompt group.

For every selected layer/group feature, it:

1. Replayed the unmodified SDXL Turbo sample.
2. Encoded the output-minus-input residual at the selected U-Net block.
3. Saved the feature's spatial activation map.
4. Generated nine matched images, one for each signed multiplier.
5. Saved a JET activation gallery, a signed image sweep, and a signed
   coefficient-overlay gallery.

The source weights were verified against the SHA-256 hashes recorded by the
source run. The spatial face/global pooled activations matched the source
`.npz` data with maximum absolute discrepancy `3.81e-06`.

## How the signed edit works

Let `z_f` be the original nonnegative SAE activation for selected feature `f`,
and `d_f` be its decoder direction. We edit the original block output directly:

```text
edited output = original output + (s - 1) × z_f × d_f
```

where `s` is the displayed multiplier.

| Multiplier | Meaning |
|---:|---|
| `1` | Original output: exact no-op control |
| `0` | Remove that feature's original contribution |
| `2`, `3`, `5` | Amplify the existing contribution |
| `-1`, `-2`, `-3`, `-5` | Reverse the contribution and scale its magnitude |

This is a multiplicative edit of an already active feature. It does not inject a
feature in places where the SAE encoded zero. It also preserves the SAE's
residual reconstruction error because it starts from the original U-Net output
instead of replacing the output with a full SAE reconstruction.

## How to read the JET visuals

All overlays use coarse SAE spatial cells resized by nearest-neighbor. That is
why they look pixelated: each square corresponds to one feature-map position,
not one image pixel.

### Baseline activation maps

The original feature activation is nonnegative. Its JET range is:

```text
low: blue → cyan → green → yellow → orange → red : high
```

Zero cells are transparent, so the original image shows through. The map's
numeric scale is shared across the three displayed seeds for one
layer-specific feature. Do not compare raw colors between different feature IDs
or different layers; those use different scales.

### Signed coefficient overlays

The signed gallery overlays `s × z_f` on the edited image. It shows the
coefficient intentionally applied by the intervention:

```text
negative: blue → cyan → green → yellow → orange → red : positive
```

Here green is near zero, blue is negative, and red is positive. This is **not**
a new measurement of post-edit SAE activation. It is not a causal pixel map.
It is a spatial record of where and with which signed coefficient the original
feature contribution was changed.

## Checks that passed

| Check | Result |
|---|---:|
| Original samples replayed exactly | 36 / 36 |
| `s = 1` no-op images pixel-identical | 144 / 144 |
| Selected layer/group features | 48 |
| Signed edited images | 1,296 |
| Maximum replay pixel error | 0 |
| Maximum no-op pixel error | 0 |

The full machine-readable record is in [verification.json](verification.json)
and [intervention_metrics.csv](intervention_metrics.csv).

## What the images can be used for

The galleries can now answer narrow visual questions, such as:

- Does a selected feature act on hair, skin/face appearance, clothing, an
  accessory, the background, or a mixture?
- Is its apparent effect different when it is edited in `down.2.1`, `mid.0`,
  `up.0.0`, or `up.0.1`?
- Does reversing a feature produce a symmetric or asymmetric effect?
- Does removal (`s = 0`) damage other image content?

Use [visual_review.csv](visual_review.csv) to record consistent manual review
for coverings, hair changes, clothing changes, face-detail changes, artifacts,
and preference. The columns are intentionally blank: annotation has not been
performed yet.

## What the images cannot establish yet

- A feature is not automatically a concept such as “race,” “skin color,” or a
  specific cultural attribute merely because it differs between prompts.
- Prompt-group associations are not a fairness score. The labels are generation
  conditions, not identities inferred from faces.
- A visually noticeable edit is not automatically a quality improvement.
- The current model is SDXL Turbo with one denoising step. The four rows compare
  U-Net depth **inside that one step**. They do not test early, middle, and late
  diffusion time.

## Connection to time-evolving SAE work

The related diffusion-SAE paper discussed in
[`layers/PAPER_CONNECTION.md`](../../../../PAPER_CONNECTION.md) studies
multi-step denoising and uses Grounding DINO/SAM masks to associate features
with objects before testing time-specific interventions. The current run is the
depth half of a possible layer-by-time experiment. A corresponding temporal
study would require capturing a multi-step model at actual scheduler timesteps,
validating SAE reconstruction per timestep, and evaluating spatial mask overlap
on held-out data.

## Reproduce

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python -I -u \
  layers/render_signed_feature_visuals.py \
  --seeds 150 151 152 \
  --strengths -5 -3 -2 -1 0 1 2 3 5
```

The script defaults to this source run and writes a fresh output directory only
when the target does not already contain files. The full-resolution images and
raw maps are intentionally local under `raw/`; compact review galleries and
metrics are committed to GitHub.
