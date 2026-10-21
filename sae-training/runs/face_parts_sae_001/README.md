# Mapping SAE features to facial parts

This update checks whether SAE features are spatially associated with face
parts. It is an exploratory localization analysis: it does not establish that
a feature has an exclusive semantic meaning or report a classification
accuracy.

## Results and gallery

The [complete held-out gallery](individual_part_epoch_grids/index.md) shows all
240 test portraits. Each sheet uses rows for face, eyes, eyebrows, nose, and
ears, and columns for SAE checkpoints at epochs 0, 4, 8, 12, 16, and 20. Cyan
marks the face box; yellow marks the part box. “No bbox detected” means the
automatic annotation pipeline did not provide a usable box for that part in
that image.

One feature was selected per part using validation data at epoch 20. The
selection score is the mean activation inside the annotated part box minus the
mean activation in a reference area, averaged first within race/gender
categories and then equally across categories.

| Part | Selected feature | Validation contrast | Images | Categories |
| --- | ---: | ---: | ---: | ---: |
| Face | 1664 | 9.085 | 240 | 12 |
| Eyes | 3701 | 9.502 | 101 | 11 |
| Eyebrows | 1600 | 13.499 | 37 | 9 |
| Nose | 1664 | 13.084 | 21 | 6 |
| Ears | 4858 | 12.006 | 17 | 6 |

Contrast is measured in SAE activation units, not as a percentage, accuracy,
or correlation coefficient. Higher positive values mean greater average
activation in the annotated region than in the reference area. Nose and ears
have relatively few usable examples, so those mappings are preliminary. Face
and nose both selecting feature 1664 means it scored highest for both
contrasts; it does not show that the feature exclusively represents either
part.

The held-out portraits are used to visualize the selected features after
selection; the test set was not used to choose the feature IDs. For a
quantitative held-out contrast analysis with bootstrap intervals, use the
validation script below. That is a separate evaluation from the visual gallery.

## Scripts

- [`render_all_face_parts_over_time.py`](../../render_all_face_parts_over_time.py): selects one feature per part on validation data and renders every held-out portrait across the six checkpoints.
- [`validate_face_part_features.py`](../../validate_face_part_features.py): evaluates fixed feature IDs on validation and test images, reporting held-out activation contrasts and bootstrap confidence intervals for eyes, eyebrows, nose, and ears.
- [`render_face_region_heatmaps.py`](../../render_face_region_heatmaps.py): renders top region-ranked features on selected held-out examples.

The current gallery is summarized in the [full experiment report](REPORT.md).
The feature sheets in `region_heatmaps/` show up to five held-out test seeds
per race/gender category, across the six saved SAE epochs; the independent
seed-200–299 test set is used for these multi-seed visual comparisons.

## Reproduce the gallery

From this directory, with the project Python environment and source data,
checkpoints, and residuals available at their configured paths:

```bash
python ../../render_all_face_parts_over_time.py --output individual_part_epoch_grids
```

The script writes `individual_part_epoch_grids/index.md`, the per-person
`sheets/`, `validation_selected_features.csv`, and `settings.json`.
