# SAE Feature Ablation Experiments

I keep the feature-ablation runs here. In these tests, I changed selected SAE feature contributions during generation while keeping the diffusion model fixed. My earlier necklace pilot is in [concept/tuning](../concept/tuning/README.md).

## Kimono, flower, and turban pilot

I attenuated candidate SAE features during generation while keeping the model and SAE weights frozen. I replayed each image with its original prompt and seed, then compared strengths 0 (remove the feature contribution), 0.5 (keep half), and 1 (unchanged) with a random feature. The unchanged outputs matched the source images exactly.

| Detector label | Candidate feature | Same-prompt images | Mean pixel change at strength 0, candidate / random | Visual spot check |
|---|---:|---:|---:|---|
| Turban | 1526 | 8 | 0.721 / 0.982 | Turban remained visible; candidate edit was subtle. |
| Flower | 2230 | 8 | 5.205 / 6.053 | Hair flower remained visible; face and clothing also shifted. |
| Kimono | 5046 | 4 | 3.984 / 4.178 | No clear garment removal; candidate behaved about like the random feature. |

Pixel change is mean absolute RGB difference on a 0–255 scale. It shows how much the image changed, not whether the named item disappeared. I have not manually reviewed the Florence-2 labels. The kimono test had only two positives and two negatives. I did not see reliable removal of any of the three items.

Open the [comparison grids](wardrobe_pilot/REPORT.md), [measurements](wardrobe_pilot/metrics.csv), or [sample manifest](wardrobe_pilot/manifest.csv). The source images are regenerated from the existing prompt and seed and are omitted from Git; the grids contain the results. The runnable script and Slurm job are [wardrobe_pilot.py](wardrobe_pilot.py) and [wardrobe_pilot.slurm](wardrobe_pilot.slurm).

I took the candidate IDs from a train-only feature ranking. I picked each control with a fixed random seed, matching activation size and frequency approximately. This is a small exploratory control, not a formal causal or statistical test.

## Headscarf and earring ablation

The separate [headscarf and earring pilot](accessory_ablation/REPORT.md) tests the discovery-ranked candidate features at strengths -1, -0.5, 0, 0.5, and 1, with same-prompt samples and matched random-feature controls. The runnable script is [accessory_ablation.py](accessory_ablation.py). These negative values subtract more than the feature's measured contribution; they do not retrain the SAE.

## C2C-box-masked earring ablation

The [box-masked pilot](box_masked_earring/REPORT.md) maps localized C2C earring boxes to the SAE's runtime spatial grid and compares local suppression with global suppression and a random-feature control. The runner is [box_masked_earring.py](box_masked_earring.py).

The [spatial feature search](earring_spatial_search/RANKING.md) ranks SAE features using C2C earring regions on training images, reports held-out scores, and tests the top three features in the boxes. Its runner is [earring_spatial_search.py](earring_spatial_search.py); the intervention comparisons are in [INTERVENTIONS.md](earring_spatial_search/INTERVENTIONS.md).
