# SAE features on simple-prompt images

No training performed. Feature IDs refer to the existing pretrained down.2.1 SAE.

Ranked using 720 discovery images; validation uses 240 separate images. The 240 test images are untouched.

These features distinguish generation-prompt groups. They do not yet identify head coverings or harmful bias.

| Prompt group | Top feature | Discovery contrast | Validation contrast |
|---|---:|---:|---:|
| black man | 584 | 3.111 | 3.472 |
| black woman | 3279 | 3.000 | 3.000 |
| white man | 3466 | 3.092 | 3.173 |
| white woman | 4384 | 3.544 | 3.720 |
| east asian man | 284 | 2.680 | 2.514 |
| east asian woman | 1899 | 3.316 | 3.032 |
| middle eastern man | 2359 | 3.123 | 3.299 |
| middle eastern woman | 3935 | 3.421 | 3.515 |
| indian man | 1050 | 2.912 | 2.937 |
| indian woman | 2527 | 3.110 | 3.345 |
| hispanic man | 2055 | 2.596 | 2.249 |
| hispanic woman | 3884 | 2.962 | 2.740 |

[Full per-group rankings](top_features_by_race_gender.csv)

[Gender-averaged race rankings](top_features_by_race.csv)

[Feature prevalence and active fractions](feature_prevalence.csv)

Run `sbatch backprop/sae_heatmaps.slurm` to replay discovery examples and visualize top-feature maps.

Before suppressing a feature, inspect examples where it fires on both covered and uncovered heads.
