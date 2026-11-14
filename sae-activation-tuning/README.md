# Mean SAE activations on the open-prompt dataset

The chart uses the 1,200-image dataset generated with prompts of the form `a picture of a {race} {gender}`. It uses the existing pretrained SAE's spatial-mean activations (`X_mean`), pools both gender groups within each race, and ranks the top ten features separately for each race using all 200 images in that race group.

Each panel shows one race's top ten feature IDs, with grouped bars showing each feature's mean activation across all six race groups. Feature IDs are shared across panels and refer to the pretrained SAE. These are descriptive prompt-group associations; they do not establish semantic meaning or causation.

The 600-image engineered neutral-expression, T-shirt, and background dataset was not used.

Files: `biased_top10_mean_activations_by_race.png` and `biased_top10_mean_activations_by_race.csv`.

The activation cache includes 720 train, 240 validation, and 240 test images. This exploratory ranking uses all three splits.

For visual starting points from the completed activation multiplier sweeps, see [reasonable multiplier candidates](tuning_activations/REASONABLE_MULTIPLIERS.md). The linked grids mark candidate tiles in red; these are preliminary visual picks from one representative seed per feature.
