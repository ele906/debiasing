# Feature 4338 — upper-face-only signed intervention

This experiment tests whether feature 4338 becomes a cleaner upper-face/head-appearance control when the SAE edit is restricted to the upper third of the automatic `human face` box.

## Design

- **Dataset:** biased/simple-prompt portraits, held-out seeds 20–29, all 12 prompt categories.
- **Model:** face-focused mid.0 SAE; diffusion model frozen.
- **Spatial mask:** upper vertical third of the automatic face box.
- **Scales:** `−7, −3, −1, 0, +1, +3, +7`; `+1` is the no-op control.
- **Coverage:** 12 × 10 × 7 = 840 variants.

## Start here

The clearest signed comparison is shown by gender. Each row is one held-out seed, selected for largest change at either −7 or +7:

- [All men: original | −7 | +7 | difference](all_men_original_neg7_pos7_difference.png)
- [All women: original | −7 | +7 | difference](all_women_original_neg7_pos7_difference.png)

These pages are visual screening aids; the selected seed is not an average-effect estimate.

## Metrics and representative overlay grids

- [Mean metrics by category and alpha](mean_metrics_by_group_alpha.csv)
- [Per-image metrics](metrics.csv)
- [Exact run settings](settings.json)

The `plain_grids`, `signed_heatmaps`, `post_activation_heatmaps`, and `footprint_heatmaps` folders contain seed-20 examples for all 12 categories. The main interpretation question is whether upper-face-only editing produces a more localized effect than global feature 4338 scaling while preserving the rest of the face.
