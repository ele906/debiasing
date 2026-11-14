# Final-checkpoint feature-to-region maps

These maps show candidate SAE feature activations on held-out images from the
independent test set (seeds 200–299). They use the selected fine-tuned SAE
checkpoint and do not compare epochs. Feature IDs were ranked using validation
data; one usable test image per prompt category is shown when available.

Each sheet has one row per race/gender prompt category. The image is overlaid
with the feature's pixelated activation map across the full image. Cyan marks
the automatic face box and yellow marks the automatic target-region box. Seed
numbers are printed beside each category. The same image appears once because
these are final-checkpoint localization views, not training-progress sheets.

The folder name (`eyes`, `eyebrows`, `nose`, or `ears`) identifies the region
being evaluated. A filename such as `feature_3593.jpg` identifies an SAE
coordinate. It is a candidate with higher validation activation contrast for
that region, not a human label or proof that the feature represents that part.
Heatmap colors are normalized per image, so use them to inspect location, not
to compare activation strength across different rows.

## Candidate maps

### Eyes

- [Feature 3701](eyes/feature_3701.jpg)
- [Feature 3593](eyes/feature_3593.jpg)
- [Feature 1600](eyes/feature_1600.jpg)

### Eyebrows

- [Feature 1237](eyebrows/feature_1237.jpg)
- [Feature 1600](eyebrows/feature_1600.jpg)
- [Feature 3701](eyebrows/feature_3701.jpg)

### Nose

- [Feature 1664](nose/feature_1664.jpg)
- [Feature 3593](nose/feature_3593.jpg)
- [Feature 1991](nose/feature_1991.jpg)

### Ears

- [Feature 4858](ears/feature_4858.jpg)
- [Feature 3725](ears/feature_3725.jpg)
- [Feature 2230](ears/feature_2230.jpg)

The epoch-by-epoch comparisons are separate: see
[`region_heatmaps/`](../region_heatmaps/) for feature-first sheets at epochs
0, 4, 8, 12, 16, and 20, or
[`individual_part_epoch_grids/`](../../individual_part_epoch_grids/index.md)
for one portrait per sheet with facial-region rows across those checkpoints.

See `samples.csv` for the test image and seed used in each row, and
`settings.json` for the checkpoint and rendering details.
