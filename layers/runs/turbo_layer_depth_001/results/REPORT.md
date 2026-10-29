# Layerwise SAE group-comparison pilot

## What we found

All 48 discovery-selected layer/group features had positive held-out mean contrasts; their seed-bootstrap 95% intervals excluded zero. These intervals are unadjusted across the 48 comparisons, and the discovery procedure intentionally selects the largest positive feature contrast, so the counts are evidence of reproducible prompt-group associations, not semantic labels.

The mean standardized contrast across the 12 prompt groups was 2.13 feature SDs in down block 2, 1.23 in the middle block, 1.41 in up block 0 attention 0, and 2.00 in up block 0 attention 1. This is not a smooth broad-to-narrow progression. For the Indian-woman prompt cell, the selected feature IDs were 3725, 3018, 4099, and 1243 across those layers, respectively. The layer-specific SAEs learn different dictionaries, so those IDs are not interchangeable, and the group association alone does not tell us whether a feature detects skin, jewelry, pose, clothing, or another correlated cue.

## Design

The same 1,200 SDXL Turbo portraits (six race prompt categories × two gender prompt categories × seeds 100–199) were encoded with four pretrained, layer-specific SAEs. For each target group and gender, discovery seeds 100–149 selected the single SAE feature with the largest mean face-box activation contrast against the other five race-prompt groups of the same gender. Held-out seeds 150–199 evaluated that fixed feature. Confidence intervals resample paired seed-level contrasts.

The heatmap uses a JET palette. X is the prompt group; Y is the model layer/SAE. Cell text gives the standardized held-out contrast (raw target-minus-reference contrast divided by the held-out SD of that layer's selected feature across same-gender samples) and that layer's feature ID. Feature IDs are specific to each SAE and must not be compared across layers as if they had the same meaning. Standardization makes the displayed contrast units more comparable across layers, but the independently trained SAEs may still encode different concepts. Two samples lacked usable face boxes and are excluded from the relevant seed-level contrast when they occur.

## Interpretation limits

This is a pilot association screen, not evidence that an SAE feature represents race, skin color, jewelry, or another human attribute. The generated prompt groups may differ in correlated visual details; automatic face boxes and face-region mean pooling can miss small/local attributes. A feature chosen as the strongest discovery contrast may be noisy, so interpret its held-out effect and interval, not the discovery score alone. The 95% intervals are unadjusted for the 48 layer/group comparisons. The SAEs in this pilot are pretrained for each layer; this does not yet test fine-tuning an SAE on this project’s face dataset.

## Files

- `layerwise_group_effects.csv`: selected feature and held-out statistics for each layer/group cell.
- `layerwise_group_effects_jet.png`: held-out comparison plot.
