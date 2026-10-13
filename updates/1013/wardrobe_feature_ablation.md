# Turban, Flower, and Kimono SAE Feature Ablation

## Question

When a candidate SAE feature is attenuated during generation, does the named item disappear while the rest of the image stays intact?

This was a small intervention pilot. It did not train or fine-tune the diffusion model or SAE.

## Images and controls

We replayed 20 same-prompt validation images using each image's original prompt and seed. The pretrained diffusion model and SAE were frozen. Samples were selected using unreviewed Florence-2 concept labels: eight turban images, eight flower images, and four kimono images. Each concept's sample included detector positives and negatives; kimono had only two of each because there were few same-prompt validation examples.

Candidate features were fixed from the train-only feature ranking. For comparison, each concept also had a random feature selected to have roughly similar activation size and frequency within this small sample.

| Concept | Candidate feature | Random control | Images | Candidate / control MAE at strength 0 |
|---|---:|---:|---:|---:|
| Turban | 1526 | 2430 | 8 | 0.721 / 0.982 |
| Flower | 2230 | 2940 | 8 | 5.205 / 6.053 |
| Kimono | 5046 | 4544 | 4 | 3.984 / 4.178 |

## Intervention

At the selected U-Net block, we scaled a candidate feature's SAE contribution while keeping the rest of the model fixed. Strength 1 was the unchanged condition, 0 removed that feature contribution, and 0.5 kept half. The same intervention was applied to the matched random feature.

All 20 source images replayed with exact pixel identity. All 40 strength-1 outputs were exact no-ops.

## Results

Visual spot checks did not show clear removal of any named item:

- **Turban, feature 1526:** the edit was subtle and the turban remained visible.
- **Flower, feature 2230:** the hair flower remained visible; the face and clothing also shifted.
- **Kimono, feature 5046:** no clear garment removal; its pixel change was close to the random-feature control.

The reported mean absolute RGB difference (MAE, 0–255 scale) measures image change, not object disappearance. The candidate changes were not stronger than their random controls on average. These results are consistent with weak or mixed features, not reliable concept-specific object control.

Labels were automatically generated and not manually reviewed; detector negatives may include missed items. The kimono sample is especially small. This pilot is exploratory and does not support claims about whether these items are desirable or whether their removal improves fairness.

## Files

- [Full report and comparison grids](../../tuning/wardrobe_pilot/REPORT.md)
- [Per-image measurements](../../tuning/wardrobe_pilot/metrics.csv)
- [Sample manifest](../../tuning/wardrobe_pilot/manifest.csv)
- [Runner](../../tuning/wardrobe_pilot.py) and [Slurm job](../../tuning/wardrobe_pilot.slurm)
- Example grids: [turban](../../tuning/wardrobe_pilot/grids/middle_eastern_man_seed095.jpg), [flower](../../tuning/wardrobe_pilot/grids/east_asian_woman_seed041.jpg), [kimono](../../tuning/wardrobe_pilot/grids/east_asian_man_seed022.jpg)
