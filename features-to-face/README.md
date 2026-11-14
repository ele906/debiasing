# SAE features to facial parts

This folder contains the **fixed-feature held-out localization analysis**. It
tests whether 19 SAE features chosen from feature-intervention runs activate
more inside automatically annotated facial-part boxes than in a reference
area. It uses the original pretrained SAE, held fixed for evaluation; there
is no fine-tuning or epoch comparison in this analysis.

## Held-out results

The candidates were fixed without looking at the held-out test activations.
For each feature and region, the score is mean SAE activation in the part box
minus activation in the reference area, macro-averaged across race/gender
categories. Intervals are 95% stratified image-bootstrap intervals.

| Region | Notable feature | Test contrast (95% interval) | Test images |
| --- | ---: | ---: | ---: |
| Eyes | 527 | +1.75 [1.40, 2.11] | 90 |
| Eyebrows | 527 | +2.66 [2.25, 3.04] | 28 |
| Nose | 1664 | +12.04 [11.48, 12.56] | 13 |
| Ears | 527 | +5.60 [4.70, 6.49] | 18 |

These are activation-unit differences, not percentages or accuracy. Feature
527 activates more in several regions, so it is not specific to one part.
Feature 1664 has the clearest nose association in this analysis: its nose
contrast is strongly positive while its eyebrow contrast is negative
(−1.51 [−2.84, −0.20]). The nose and ear estimates use small test sets, and
the automatic boxes have not been manually reviewed. Treat these results as
exploratory evidence of spatial association, not proof of semantic meaning or
causal editing behavior.

## Files

- [`RESULTS.md`](RESULTS.md): full interpretation, setup, and selected held-out heatmap examples.
- [`heldout_feature_validation.csv`](heldout_feature_validation.csv): estimates for all 19 features across all four parts, including confidence intervals and coverage.
- [`settings.json`](settings.json): original SAE checkpoint, fixed candidate IDs, box filters, and test-set coverage.
- [`images/feature_part_examples.jpg`](images/feature_part_examples.jpg): illustrative held-out overlays. Each example was chosen for high target-box activation, so these panels are not representative samples.
- [`seed_grids/index.md`](seed_grids/index.md): full per-feature, per-part visual grids across 12 categories.

## Scripts

- [`validate_face_part_features.py`](../sae-training/validate_face_part_features.py): computes the region-versus-reference activation contrasts and bootstrap intervals on validation and test splits.
- [`render_feature_part_examples.py`](../sae-training/render_feature_part_examples.py): renders the selected heatmap examples with face and part boxes.
- [`build_feature_part_grids.py`](build_feature_part_grids.py): renders a separate heatmap-and-box grid for every feature/part pair using validation and test portraits.

The [fine-tuning epoch gallery](../sae-training/runs/face_parts_sae_001/individual_part_epoch_grids/index.md) is a separate analysis of a fine-tuned SAE. It should not be confused with this fixed-pretrained-SAE feature-to-part test.
