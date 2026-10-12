# SAE feature tuning pilots

This folder holds the newest intervention experiment. The earlier necklace pilot remains documented in [concept/tuning](../concept/tuning/README.md).

## Kimono, flower, and turban pilot

We attenuated candidate SAE features in the diffusion model while keeping the model and SAE weights frozen. Each source image was replayed with its original prompt and seed. For each concept, we compared feature strength 0 (remove its SAE contribution), 0.5 (keep half), and 1 (unchanged) against a random feature with similar activation. The unchanged outputs reproduced the source images exactly.

| Detector label | Candidate feature | Same-prompt images | Mean pixel change at strength 0, candidate / random | Visual spot check |
|---|---:|---:|---:|---|
| Turban | 1526 | 8 | 0.721 / 0.982 | Turban remained visible; candidate edit was subtle. |
| Flower | 2230 | 8 | 5.205 / 6.053 | Hair flower remained visible; face and clothing also shifted. |
| Kimono | 5046 | 4 | 3.984 / 4.178 | No clear garment removal; candidate behaved about like the random feature. |

Pixel change is mean absolute RGB difference on a 0–255 scale. It measures how much the image changed, not whether the named object disappeared. The Florence-2 labels are unreviewed. Kimono has only two positive and two negative examples because the original validation set lacks enough same-prompt positives. Treat all three as weak exploratory tests; none shows reliable concept-specific removal.

Open the [comparison grids](wardrobe_pilot/REPORT.md), [measurements](wardrobe_pilot/metrics.csv), or [sample manifest](wardrobe_pilot/manifest.csv). The source images are regenerated from the existing prompt and seed and are omitted from Git; the grids contain the results. The runnable script and Slurm job are [wardrobe_pilot.py](wardrobe_pilot.py) and [wardrobe_pilot.slurm](wardrobe_pilot.slurm).

The fixed candidate IDs came from the train-only feature ranking. The control feature was sampled with a fixed random seed from features matched approximately on activation size and frequency within this selected set. This small control is informative but is not a formal causal or statistical test.

## Headscarf and earring ablation

The separate [headscarf and earring pilot](accessory_ablation/REPORT.md) tests the discovery-ranked candidate features at strengths -1, -0.5, 0, 0.5, and 1, with same-prompt samples and matched random-feature controls. The runnable script is [accessory_ablation.py](accessory_ablation.py). These negative values subtract more than the feature's measured contribution; they do not retrain the SAE.

## C2C-box-masked earring ablation

The [box-masked pilot](box_masked_earring/REPORT.md) maps localized C2C earring boxes to the SAE's runtime spatial grid and compares local suppression with global suppression and a random-feature control. The runner is [box_masked_earring.py](box_masked_earring.py).
