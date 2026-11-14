# Face-focused mid.0 SAE — feature 1874 intervention

This report records the completed intervention sweep for **feature 1874** in the **face-box fine-tuned SAE** at `unet.mid_block.attentions.0` (mid.0).  The feature was chosen as the strongest validation-set face-vs-outside feature: it is more active inside automatically detected face boxes than outside them.  This experiment tests what happens when we directly scale that one SAE feature during generation; it does **not** claim that the feature represents a single human attribute.

## What was run

- **Model:** the face-box fine-tuned mid.0 SAE (selected epoch 20), inserted into the frozen diffusion model.
- **Coverage:** 12 race/gender prompt categories × 20 fixed seeds × 13 intervention strengths = **3,120 generated variants**.
- **Strengths:** `-10, -7, -5, -3, -2, -1, 0, +1, +2, +3, +5, +7, +10`.
- **Measurements:** post-re-encoded feature activation, active spatial fraction, RGB pixel MAE from the original, and SAE-coefficient change magnitudes.  The category-level results are in [mean_metrics_by_group_alpha.csv](mean_metrics_by_group_alpha.csv).

## How to read the grids

Each representative sheet is a **3 × 5** grid for seed 0.  The original generation is at the top-left.  The rest are in ascending alpha order:

| Original | −10 | −7 | −5 | −3 |
|---|---:|---:|---:|---:|
| −2 | −1 | 0 | +1 | +2 |
| +3 | +5 | +7 | +10 | — |

Positive alpha boosts feature 1874 and negative alpha attenuates it.  The signed-overlay counterpart visualizes where the SAE coefficient changed (cool colors negative, warm colors positive).  One seed is included here for quick inspection; the full local run contains all 20 seeds per category and the per-image data.

## Representative seed-0 sheets and activation curves

| Category | Generated sweep | Signed coefficient overlay | Mean activation curve |
|---|---|---|---|
| Black man | [grid](representative_grids/black_man_seed0.jpg) | [overlay](signed_overlays/black_man_seed0.jpg) | [curve](activation_graphs/black_man.png) |
| Black woman | [grid](representative_grids/black_woman_seed0.jpg) | [overlay](signed_overlays/black_woman_seed0.jpg) | [curve](activation_graphs/black_woman.png) |
| White man | [grid](representative_grids/white_man_seed0.jpg) | [overlay](signed_overlays/white_man_seed0.jpg) | [curve](activation_graphs/white_man.png) |
| White woman | [grid](representative_grids/white_woman_seed0.jpg) | [overlay](signed_overlays/white_woman_seed0.jpg) | [curve](activation_graphs/white_woman.png) |
| East Asian man | [grid](representative_grids/east_asian_man_seed0.jpg) | [overlay](signed_overlays/east_asian_man_seed0.jpg) | [curve](activation_graphs/east_asian_man.png) |
| East Asian woman | [grid](representative_grids/east_asian_woman_seed0.jpg) | [overlay](signed_overlays/east_asian_woman_seed0.jpg) | [curve](activation_graphs/east_asian_woman.png) |
| Middle Eastern man | [grid](representative_grids/middle_eastern_man_seed0.jpg) | [overlay](signed_overlays/middle_eastern_man_seed0.jpg) | [curve](activation_graphs/middle_eastern_man.png) |
| Middle Eastern woman | [grid](representative_grids/middle_eastern_woman_seed0.jpg) | [overlay](signed_overlays/middle_eastern_woman_seed0.jpg) | [curve](activation_graphs/middle_eastern_woman.png) |
| Indian man | [grid](representative_grids/indian_man_seed0.jpg) | [overlay](signed_overlays/indian_man_seed0.jpg) | [curve](activation_graphs/indian_man.png) |
| Indian woman | [grid](representative_grids/indian_woman_seed0.jpg) | [overlay](signed_overlays/indian_woman_seed0.jpg) | [curve](activation_graphs/indian_woman.png) |
| Hispanic man | [grid](representative_grids/hispanic_man_seed0.jpg) | [overlay](signed_overlays/hispanic_man_seed0.jpg) | [curve](activation_graphs/hispanic_man.png) |
| Hispanic woman | [grid](representative_grids/hispanic_woman_seed0.jpg) | [overlay](signed_overlays/hispanic_woman_seed0.jpg) | [curve](activation_graphs/hispanic_woman.png) |

## Interpretation and next step

This is a controlled *causal intervention* test: prompt and seed stay fixed within each row of a sheet, while only the SAE coefficient scale changes.  The next analysis step is blinded visual review across many seeds, then selecting the smallest alpha range that reliably yields a coherent, face-localized change with limited collateral change.  Stronger causal evidence should come from repeating the sweep on an independent prompt/seed set rather than interpreting a single gallery as proof of semantics.

Run configuration: [settings.json](settings.json).
