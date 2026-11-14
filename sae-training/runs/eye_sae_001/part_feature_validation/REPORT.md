# Held-out facial-part feature validation

## Result

The candidates selected without using test data all had positive mean
activation contrast on the held-out test images. The interval shown is a 95%
stratified image-bootstrap interval, with images resampled within each
race/gender category and then averaged across categories.

| Region | Feature IDs | Test images / categories | Test contrast (95% interval) |
|---|---|---:|---|
| Eyes | 3701, 3593, 1600 | 90 / 9 | 9.03 [7.27, 10.66]; 7.94 [6.77, 9.02]; 6.80 [6.30, 7.31] |
| Eyebrows | 1600, 3701, 1237 | 28 / 7 | 7.81 [6.96, 8.62]; 7.96 [6.07, 9.70]; 6.27 [4.27, 8.25] |
| Nose | 1664, 3593, 1991 | 13 / 4 | 11.79 [11.26, 12.26]; 9.70 [7.86, 11.49]; 6.54 [5.08, 8.08] |
| Ears | 4858, 2230, 2611 | 18 / 6 | 10.18 [8.98, 11.40]; 9.46 [7.52, 11.37]; 5.39 [2.43, 8.13] |

The values are differences in mean SAE activation (annotated part minus
reference area); they are not percentages or reconstruction scores. Eyes,
eyebrows, and nose are compared with other cells inside the face box. Ears are
compared with the rest of the image because ear boxes can extend beyond the
detected face. Nose coverage is especially sparse (13 test images across four
categories); both nose and ear estimates are preliminary.

## How to interpret this

All reported intervals are above zero, which is evidence that these features
activate more in the annotated areas than in their reference areas on these
held-out samples. This supports spatial association with the automatic
annotations. It does not establish that a feature semantically represents an
eye, eyebrow, or ear, nor that it is specific to that part: several candidates
are shared across regions, and the ear reference includes background and other
image content. The boxes are automatic and unreviewed; visual inspection and a
larger, reviewed ear set would strengthen the conclusion.

## Method and files

Eye candidates were taken from the eye fine-tuning run's validation ranking.
Eyebrow, nose, and ear candidates were ranked by macro-average category
activation contrast on validation data. Candidate selection did not use test
activations.
Region boxes covering more than 12% of the image were excluded, as were
examples with fewer than two region cells or reference cells. Earrings were
excluded from ear annotations.

- `heldout_feature_validation.csv` — per-feature validation and test contrast,
  bootstrap interval, activation means, image counts, and category counts.
- `settings.json` — model, selection/evaluation protocol, box filters, and
  sample counts.
- `../eye_localized_features.csv` — validation ranking used for the eye
  candidates.
