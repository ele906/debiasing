# Biased-dataset held-out test — face-focused mid.0 SAE feature 1874

This is the corrected independent evaluation on the **biased dataset**.  It keeps the same biased/simple prompt template and uses seeds that were not included in the earlier intervention sweep.

## Design

- **Model:** face-box fine-tuned SAE at `unet.mid_block.attentions.0` (mid.0); the diffusion model remains frozen.
- **Feature:** 1874, selected previously as a face-localized SAE feature.
- **Dataset:** the original biased/simple-prompt dataset only (`a picture of a {race} {gender}`).
- **Held-out split:** seeds **20–29** in every one of the 12 prompt categories.  The first sweep used seeds 0–19.
- **Scale values:** `−7, −5, −3, −2, −1, 0, +1, +2, +3, +5, +7`; `+1` is the no-op control.
- **Coverage:** 12 categories × 10 unused seeds × 11 scales = **1,320 generated variants**.

This is the result that should be compared directly with the first biased-dataset sweep.  The earlier neutral-prompt run is separate supplementary work and does not replace this biased-dataset test.

## Files

- [Mean metrics by category and alpha](mean_metrics_by_group_alpha.csv)
- [Per-image metrics](metrics.csv)
- [Exact settings](settings.json)

## How to read each 3 × 4 grid

Plain and heatmap grids use the same tile order:

| Original | −7 | −5 | −3 |
|---|---:|---:|---:|
| −2 | −1 | 0 | +1 control |
| +2 | +3 | +5 | +7 |

There are three overlay versions:

1. **Signed-change heatmap:** where the edited SAE coefficient moves; green→blue→purple is negative, yellow→orange→red is positive.
2. **Post-activation heatmap:** where feature 1874 is active after the intervention.
3. **Intervention-footprint heatmap:** magnitude of the direct coefficient perturbation.

Each tile has its own 99.5th-percentile display scale so spatial locations are visible.  This helps qualitative review but should not be used to compare color intensity across different tiles.

## Representative held-out seed 20

| Category | Plain grid | Signed-change heatmap | Post-activation heatmap | Footprint heatmap | Curve |
|---|---|---|---|---|---|
| Black man | [plain](plain_grids/black_man_seed20.jpg) | [signed](signed_change_heatmaps/black_man_seed20.jpg) | [post](post_activation_heatmaps/black_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/black_man_seed20.jpg) | [curve](activation_graphs/black_man.png) |
| Black woman | [plain](plain_grids/black_woman_seed20.jpg) | [signed](signed_change_heatmaps/black_woman_seed20.jpg) | [post](post_activation_heatmaps/black_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/black_woman_seed20.jpg) | [curve](activation_graphs/black_woman.png) |
| White man | [plain](plain_grids/white_man_seed20.jpg) | [signed](signed_change_heatmaps/white_man_seed20.jpg) | [post](post_activation_heatmaps/white_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/white_man_seed20.jpg) | [curve](activation_graphs/white_man.png) |
| White woman | [plain](plain_grids/white_woman_seed20.jpg) | [signed](signed_change_heatmaps/white_woman_seed20.jpg) | [post](post_activation_heatmaps/white_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/white_woman_seed20.jpg) | [curve](activation_graphs/white_woman.png) |
| East Asian man | [plain](plain_grids/east_asian_man_seed20.jpg) | [signed](signed_change_heatmaps/east_asian_man_seed20.jpg) | [post](post_activation_heatmaps/east_asian_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/east_asian_man_seed20.jpg) | [curve](activation_graphs/east_asian_man.png) |
| East Asian woman | [plain](plain_grids/east_asian_woman_seed20.jpg) | [signed](signed_change_heatmaps/east_asian_woman_seed20.jpg) | [post](post_activation_heatmaps/east_asian_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/east_asian_woman_seed20.jpg) | [curve](activation_graphs/east_asian_woman.png) |
| Middle Eastern man | [plain](plain_grids/middle_eastern_man_seed20.jpg) | [signed](signed_change_heatmaps/middle_eastern_man_seed20.jpg) | [post](post_activation_heatmaps/middle_eastern_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/middle_eastern_man_seed20.jpg) | [curve](activation_graphs/middle_eastern_man.png) |
| Middle Eastern woman | [plain](plain_grids/middle_eastern_woman_seed20.jpg) | [signed](signed_change_heatmaps/middle_eastern_woman_seed20.jpg) | [post](post_activation_heatmaps/middle_eastern_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/middle_eastern_woman_seed20.jpg) | [curve](activation_graphs/middle_eastern_woman.png) |
| Indian man | [plain](plain_grids/indian_man_seed20.jpg) | [signed](signed_change_heatmaps/indian_man_seed20.jpg) | [post](post_activation_heatmaps/indian_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/indian_man_seed20.jpg) | [curve](activation_graphs/indian_man.png) |
| Indian woman | [plain](plain_grids/indian_woman_seed20.jpg) | [signed](signed_change_heatmaps/indian_woman_seed20.jpg) | [post](post_activation_heatmaps/indian_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/indian_woman_seed20.jpg) | [curve](activation_graphs/indian_woman.png) |
| Hispanic man | [plain](plain_grids/hispanic_man_seed20.jpg) | [signed](signed_change_heatmaps/hispanic_man_seed20.jpg) | [post](post_activation_heatmaps/hispanic_man_seed20.jpg) | [footprint](intervention_footprint_heatmaps/hispanic_man_seed20.jpg) | [curve](activation_graphs/hispanic_man.png) |
| Hispanic woman | [plain](plain_grids/hispanic_woman_seed20.jpg) | [signed](signed_change_heatmaps/hispanic_woman_seed20.jpg) | [post](post_activation_heatmaps/hispanic_woman_seed20.jpg) | [footprint](intervention_footprint_heatmaps/hispanic_woman_seed20.jpg) | [curve](activation_graphs/hispanic_woman.png) |

## Next interpretation step

The causal manipulation now generalizes to unused **biased-dataset** seeds.  Review the plain grids alongside the signed-change and footprint heatmaps: the key question is whether visible edits and their spatial footprint stay concentrated in the face rather than spilling into hair, clothing, or background.  That supports face localization; it does not by itself identify a specific facial-part semantics.
