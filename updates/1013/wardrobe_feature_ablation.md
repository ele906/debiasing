# Turban, Flower, and Kimono SAE Feature Ablation

## What I tested

I wanted to see whether reducing a candidate SAE feature during generation would remove a turban, flower, or kimono while leaving the rest of the image alone. This was a small ablation pilot; I did not train the diffusion model or SAE.

## Images and controls

I replayed 20 validation images with their original prompts and seeds, keeping the diffusion model and pretrained SAE frozen. I used unreviewed Florence-2 labels to select eight turban images, eight flower images, and four kimono images. The kimono set had only two detector positives and two negatives because there were few same-prompt examples.

I took the candidate features from the train-only ranking. For each item, I also tested a random feature chosen to have roughly similar activation size and frequency in this small sample.

| Concept | Candidate feature | Random control | Images | Candidate / control MAE at strength 0 |
|---|---:|---:|---:|---:|
| Turban | 1526 | 2430 | 8 | 0.721 / 0.982 |
| Flower | 2230 | 2940 | 8 | 5.205 / 6.053 |
| Kimono | 5046 | 4544 | 4 | 3.984 / 4.178 |

## Intervention

At one U-Net block, I scaled a candidate feature's contribution while keeping the model weights fixed. Strength 1 was unchanged, 0 removed the contribution, and 0.5 kept half. I applied the same strengths to the random feature.

All 20 source images replayed pixel-for-pixel, and all 40 strength-1 outputs were exact no-ops.

## Results

I checked the comparison grids. None of the three items clearly disappeared:

- **Turban, feature 1526:** the edit was subtle and the turban remained visible.
- **Flower, feature 2230:** the hair flower remained visible; the face and clothing also shifted.
- **Kimono, feature 5046:** no clear garment removal; its pixel change was close to the random-feature control.

The MAE is average RGB pixel change on a 0–255 scale. It measures image change, not whether an item disappeared. The candidate features changed the images about as much as, or less than, the random controls.

I did not manually review the automatic labels, so some negatives may contain the item. The kimono sample is especially small. I would treat this as an exploratory test, not evidence about whether these items are desirable or whether removing them improves fairness.

## Files

- [Full report and comparison grids](../../tuning/wardrobe_pilot/REPORT.md)
- [Per-image measurements](../../tuning/wardrobe_pilot/metrics.csv)
- [Sample manifest](../../tuning/wardrobe_pilot/manifest.csv)
- [Runner](../../tuning/wardrobe_pilot.py) and [Slurm job](../../tuning/wardrobe_pilot.slurm)
- Example grids: [turban](../../tuning/wardrobe_pilot/grids/middle_eastern_man_seed095.jpg), [flower](../../tuning/wardrobe_pilot/grids/east_asian_woman_seed041.jpg), [kimono](../../tuning/wardrobe_pilot/grids/east_asian_man_seed022.jpg)
