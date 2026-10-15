# Feature 527 signed-weight interventions

Each PNG is a single-row comparison: original, then SAE feature weight -1, -0.5, 0, and 0.5. The run used the pretrained SAE at `unet.down_blocks.2.attentions.1`, 12 prompt-label categories, and seeds 40–42. Each feature was intervened on independently while preserving the SAE reconstruction error. The 36 replayed originals matched the saved baselines exactly.

Weights are multiplicative factors on the feature contribution: 0 removes it, -1 reverses it, and 0.5 retains half. Pixel change is mean absolute RGB difference from the matched original, on a 0–255 scale; it measures image change, not quality, semantic control, or fairness.

| Feature weight | Mean pixel change | Changed samples |
|---:|---:|---:|
| -1.0 | 4.72 | 24/36 |
| -0.5 | 3.53 | 24/36 |
| 0.0 | 2.33 | 24/36 |
| 0.5 | 1.34 | 24/36 |

## Comparison strips

- [seed40_feature527_black_man_weights.png](./seed40_feature527_black_man_weights.png)
- [seed40_feature527_black_woman_weights.png](./seed40_feature527_black_woman_weights.png)
- [seed40_feature527_east_asian_man_weights.png](./seed40_feature527_east_asian_man_weights.png)
- [seed40_feature527_east_asian_woman_weights.png](./seed40_feature527_east_asian_woman_weights.png)
- [seed40_feature527_hispanic_man_weights.png](./seed40_feature527_hispanic_man_weights.png)
- [seed40_feature527_hispanic_woman_weights.png](./seed40_feature527_hispanic_woman_weights.png)
- [seed40_feature527_indian_man_weights.png](./seed40_feature527_indian_man_weights.png)
- [seed40_feature527_indian_woman_weights.png](./seed40_feature527_indian_woman_weights.png)
- [seed40_feature527_middle_eastern_man_weights.png](./seed40_feature527_middle_eastern_man_weights.png)
- [seed40_feature527_middle_eastern_woman_weights.png](./seed40_feature527_middle_eastern_woman_weights.png)
- [seed40_feature527_white_man_weights.png](./seed40_feature527_white_man_weights.png)
- [seed40_feature527_white_woman_weights.png](./seed40_feature527_white_woman_weights.png)
- [seed41_feature527_black_man_weights.png](./seed41_feature527_black_man_weights.png)
- [seed41_feature527_black_woman_weights.png](./seed41_feature527_black_woman_weights.png)
- [seed41_feature527_east_asian_man_weights.png](./seed41_feature527_east_asian_man_weights.png)
- [seed41_feature527_east_asian_woman_weights.png](./seed41_feature527_east_asian_woman_weights.png)
- [seed41_feature527_hispanic_man_weights.png](./seed41_feature527_hispanic_man_weights.png)
- [seed41_feature527_hispanic_woman_weights.png](./seed41_feature527_hispanic_woman_weights.png)
- [seed41_feature527_indian_man_weights.png](./seed41_feature527_indian_man_weights.png)
- [seed41_feature527_indian_woman_weights.png](./seed41_feature527_indian_woman_weights.png)
- [seed41_feature527_middle_eastern_man_weights.png](./seed41_feature527_middle_eastern_man_weights.png)
- [seed41_feature527_middle_eastern_woman_weights.png](./seed41_feature527_middle_eastern_woman_weights.png)
- [seed41_feature527_white_man_weights.png](./seed41_feature527_white_man_weights.png)
- [seed41_feature527_white_woman_weights.png](./seed41_feature527_white_woman_weights.png)
- [seed42_feature527_black_man_weights.png](./seed42_feature527_black_man_weights.png)
- [seed42_feature527_black_woman_weights.png](./seed42_feature527_black_woman_weights.png)
- [seed42_feature527_east_asian_man_weights.png](./seed42_feature527_east_asian_man_weights.png)
- [seed42_feature527_east_asian_woman_weights.png](./seed42_feature527_east_asian_woman_weights.png)
- [seed42_feature527_hispanic_man_weights.png](./seed42_feature527_hispanic_man_weights.png)
- [seed42_feature527_hispanic_woman_weights.png](./seed42_feature527_hispanic_woman_weights.png)
- [seed42_feature527_indian_man_weights.png](./seed42_feature527_indian_man_weights.png)
- [seed42_feature527_indian_woman_weights.png](./seed42_feature527_indian_woman_weights.png)
- [seed42_feature527_middle_eastern_man_weights.png](./seed42_feature527_middle_eastern_man_weights.png)
- [seed42_feature527_middle_eastern_woman_weights.png](./seed42_feature527_middle_eastern_woman_weights.png)
- [seed42_feature527_white_man_weights.png](./seed42_feature527_white_man_weights.png)
- [seed42_feature527_white_woman_weights.png](./seed42_feature527_white_woman_weights.png)
