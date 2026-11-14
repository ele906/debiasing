# Biased-dataset attenuation test — face-focused mid.0 SAE feature 4228

This report tests attenuation of feature 4228 in the face-box fine-tuned SAE at `unet.mid_block.attentions.0`.

## Design

- **Dataset:** biased/simple-prompt dataset only.
- **Held-out seeds:** 20–29 in each of 12 prompt categories; these were not used by the earlier feature-1874 sweep.
- **SAE:** the face-focused mid.0 SAE; the diffusion model remains frozen.
- **Scales:** `−7, −5, −3, −2, −1, 0, +1`; `+1` is unchanged and `0` removes the feature.
- **Coverage:** 12 × 10 × 7 = **840 generated variants**.

## Main quantitative result

At the `+1` control, mean feature activation was 1.130 across categories.  At zero and all negative scales, post-re-encoded activation fell to zero.  The average pixel change grew smoothly as suppression strength increased: 0.90 at alpha 0, 1.44 at −1, 2.30 at −3, and 3.51 at −7 (RGB MAE, 0–255 scale).

Feature 4228 was unevenly active across the held-out prompt groups: it was strongest in Middle Eastern man and Indian man prompts, and near zero in several woman prompt groups.  This makes it a candidate facial-hair-associated feature, but not a universal face feature; the visual grids and heatmaps should be reviewed before naming its semantics.

- [Mean metrics by category and alpha](mean_metrics_by_group_alpha.csv)
- [All per-image metrics](metrics.csv)
- [Run settings](settings.json)

## Grid order and overlays

Every grid is ordered:

| Original | −7 | −5 |
|---|---:|---:|
| −3 | −2 | −1 |
| 0 | +1 control | — |

The signed heatmap colors show coefficient direction (green→blue→purple = negative; yellow→orange→red = positive).  Post-activation heatmaps show where the feature remains active; footprint heatmaps show the direct perturbation magnitude.  Each tile is normalized separately for visualization.

## Representative held-out seed 20

| Category | Plain | Signed heatmap | Post-activation | Footprint | Curve |
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
