# Face-focused mid.0 SAE — feature screening results

This page puts the four completed comparison runs in one place.  All runs use the biased/simple-prompt dataset and the held-out seeds 20–29, across all 12 prompt categories.  The diffusion model and face-focused SAE checkpoint are fixed.

## What to look at first

Open the **plain grids** for the same category across the four rows below.  Each sheet starts with the original image, then shows the listed feature scales.  The heatmap versions help answer whether the intervention location is face-localized.

| Run | Question | Scale values | Results |
|---|---|---|---|
| Feature 4228 boost | Does boosting 4228 reveal a clearer effect? | +1, +2, +3, +5, +7 | [images and metrics](feature_4228_boost/) |
| Feature 4228 face mask | Does restricting 4228 editing to automatic face boxes reduce collateral changes? | −3, −1, 0, +1 | [images and metrics](feature_4228_face_masked/) |
| Feature 4338 screen | Is the next face-localized candidate more controllable? | −3, −1, 0, +1, +3 | [images and metrics](feature_4338_screen/) |
| Feature 3134 screen | Is the third face-localized candidate more controllable? | −3, −1, 0, +1, +3 | [images and metrics](feature_3134_screen/) |

Every subfolder contains:

- `plain_grids/`: the generated images.
- `signed_heatmaps/`: signed coefficient changes.
- `post_activation_heatmaps/`: where the selected feature is active after the edit.
- `footprint_heatmaps/`: direct edit magnitude.
- `activation_graphs/` and `mean_metrics_by_group_alpha.csv`: quantitative summaries.

## Easiest way to see feature 4228

The original grids are hard to compare. These pages instead show **original | alpha +7 | amplified absolute RGB difference** for all 12 categories.  For each category, the shown seed is the one with the largest change among held-out seeds 20–29; this makes the effect easy to inspect but is not an average-effect claim.

- [Groups 1–6: strongest-change comparison](feature_4228_boost/all_groups_original_plus7_difference_page1.png)
- [Groups 7–12: strongest-change comparison](feature_4228_boost/all_groups_original_plus7_difference_page2.png)
- [Selection details](feature_4228_boost/all_groups_selection.csv)

## Important interpretation rule

A feature is useful for controlled editing only if it has a visible, repeatable, local change at a strength where the face remains realistic.  A numerical activation change alone is not enough.  This page is intentionally a screen: its purpose is to choose one promising direction for a later feature-bundle or more precise intervention, not to claim semantics for every feature.
