# Top discovery features by layer and prompt group

This table lists the top 5 features selected using **only discovery seeds 100–149**.
The later signed image sweeps use held-out seed(s), so held-out images do not choose these IDs.

`held-out SD` is the held-out target-minus-reference activation contrast divided
by the held-out standard deviation of that feature among prompt groups of the same
gender. This makes magnitudes more comparable across features and layers. The
ranking itself still uses raw discovery contrast only; held-out values do not choose
the feature IDs. Neither quantity is a semantic label, fairness score, or quality score.

## down.2.1 (`unet.down_blocks.2.attentions.1`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---:|---:|---:|---:|---:|
| black man | F527 (+2.61 SD) | F4022 (+1.56 SD) | F2137 (+1.73 SD) | F1237 (+1.33 SD) | F1991 (+1.36 SD) |
| east asian man | F3722 (+1.33 SD) | F1759 (+2.42 SD) | F1600 (+1.09 SD) | F2808 (+1.45 SD) | F2810 (+0.67 SD) |
| hispanic man | F1653 (+1.69 SD) | F199 (+1.95 SD) | F1678 (+1.58 SD) | F2940 (+1.11 SD) | F4001 (+1.39 SD) |
| indian man | F1050 (+2.31 SD) | F3722 (+1.95 SD) | F669 (+1.19 SD) | F452 (+1.01 SD) | F1600 (+1.15 SD) |
| middle eastern man | F138 (+2.63 SD) | F452 (+1.91 SD) | F669 (+2.01 SD) | F1050 (+0.74 SD) | F4858 (+1.55 SD) |
| white man | F2707 (+2.05 SD) | F4262 (+2.28 SD) | F243 (+2.26 SD) | F2810 (+1.65 SD) | F1664 (+1.20 SD) |
| black woman | F527 (+2.49 SD) | F243 (+2.39 SD) | F3279 (+1.99 SD) | F4022 (+1.74 SD) | F2230 (+1.32 SD) |
| east asian woman | F1759 (+2.64 SD) | F2810 (+2.41 SD) | F1600 (+1.46 SD) | F2808 (+2.32 SD) | F2230 (+0.94 SD) |
| hispanic woman | F527 (+0.36 SD) | F4022 (+0.86 SD) | F1584 (+1.02 SD) | F1600 (+0.45 SD) | F199 (+1.83 SD) |
| indian woman | F3725 (+2.23 SD) | F3722 (+2.16 SD) | F4858 (+2.07 SD) | F2527 (+2.42 SD) | F1050 (+2.27 SD) |
| middle eastern woman | F138 (+2.66 SD) | F3462 (+2.33 SD) | F3701 (+1.76 SD) | F3935 (+2.60 SD) | F669 (+1.70 SD) |
| white woman | F4384 (+2.61 SD) | F472 (+2.22 SD) | F2250 (+2.16 SD) | F2611 (+1.55 SD) | F546 (+1.69 SD) |

## mid.0 (`unet.mid_block.attentions.0`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---:|---:|---:|---:|---:|
| black man | F2138 (+2.13 SD) | F2875 (+0.75 SD) | F4338 (+0.38 SD) | F3482 (+0.52 SD) | F534 (+0.61 SD) |
| east asian man | F2385 (+0.93 SD) | F1424 (+1.20 SD) | F3134 (+1.11 SD) | F4719 (+1.48 SD) | F3828 (+0.70 SD) |
| hispanic man | F4601 (+0.32 SD) | F826 (+0.55 SD) | F169 (+0.70 SD) | F1740 (+0.58 SD) | F3828 (+0.54 SD) |
| indian man | F4228 (+0.87 SD) | F2023 (+0.66 SD) | F3770 (+0.62 SD) | F3018 (+0.63 SD) | F1373 (+1.52 SD) |
| middle eastern man | F4228 (+1.87 SD) | F2023 (+1.76 SD) | F3770 (+1.71 SD) | F4399 (+1.73 SD) | F5029 (+1.17 SD) |
| white man | F4338 (+1.60 SD) | F3482 (+1.23 SD) | F1874 (+0.39 SD) | F374 (+0.99 SD) | F836 (+1.11 SD) |
| black woman | F4338 (+1.30 SD) | F1650 (+1.69 SD) | F2138 (+1.88 SD) | F4601 (+0.69 SD) | F2361 (+0.39 SD) |
| east asian woman | F1424 (+0.65 SD) | F3828 (+0.58 SD) | F846 (+1.19 SD) | F4188 (+0.84 SD) | F1740 (+1.11 SD) |
| hispanic woman | F1424 (+0.59 SD) | F3828 (+0.56 SD) | F3770 (+0.63 SD) | F247 (+0.77 SD) | F3680 (+0.47 SD) |
| indian woman | F3018 (+0.88 SD) | F1874 (+0.91 SD) | F2023 (+0.56 SD) | F1373 (+0.92 SD) | F4755 (+1.62 SD) |
| middle eastern woman | F2023 (+1.62 SD) | F5029 (+1.98 SD) | F4399 (+1.71 SD) | F3770 (+1.02 SD) | F3067 (+1.60 SD) |
| white woman | F2361 (+1.97 SD) | F5094 (+2.26 SD) | F4786 (+1.35 SD) | F1424 (+0.79 SD) | F3348 (+1.49 SD) |

## up.0.0 (`unet.up_blocks.0.attentions.0`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---:|---:|---:|---:|---:|
| black man | F2975 (+0.94 SD) | F1573 (+0.71 SD) | F3111 (+0.86 SD) | F1451 (+1.25 SD) | F2618 (+1.48 SD) |
| east asian man | F641 (+0.84 SD) | F2854 (+0.89 SD) | F573 (+1.19 SD) | F892 (+1.15 SD) | F4323 (+0.63 SD) |
| hispanic man | F1573 (+1.06 SD) | F3339 (+0.88 SD) | F3137 (+1.08 SD) | F3111 (+0.61 SD) | F1451 (+0.50 SD) |
| indian man | F2261 (+1.78 SD) | F4540 (+1.14 SD) | F3742 (+0.89 SD) | F4782 (+2.14 SD) | F2655 (+1.69 SD) |
| middle eastern man | F3742 (+1.96 SD) | F4540 (+1.67 SD) | F2261 (+1.30 SD) | F2766 (+2.32 SD) | F1467 (+2.57 SD) |
| white man | F2975 (+1.22 SD) | F1811 (+1.80 SD) | F1354 (+1.49 SD) | F3339 (+1.06 SD) | F1591 (+1.58 SD) |
| black woman | F4782 (+2.16 SD) | F4163 (+1.66 SD) | F2975 (+0.84 SD) | F2706 (+0.70 SD) | F4659 (+1.50 SD) |
| east asian woman | F4540 (+0.96 SD) | F4323 (+1.37 SD) | F4367 (+1.88 SD) | F3052 (+2.26 SD) | F2854 (+0.54 SD) |
| hispanic woman | F2706 (+0.63 SD) | F4621 (+1.01 SD) | F641 (+0.54 SD) | F3716 (+0.47 SD) | F853 (+0.27 SD) |
| indian woman | F4099 (+1.30 SD) | F4540 (+1.17 SD) | F2706 (+1.26 SD) | F1292 (+1.70 SD) | F1486 (+1.33 SD) |
| middle eastern woman | F4099 (+1.86 SD) | F1088 (+2.46 SD) | F4540 (+1.06 SD) | F608 (+2.24 SD) | F2655 (+1.78 SD) |
| white woman | F4376 (+2.19 SD) | F595 (+2.28 SD) | F1343 (+1.67 SD) | F3716 (+0.52 SD) | F641 (+0.77 SD) |

## up.0.1 (`unet.up_blocks.0.attentions.1`)

| Prompt group | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---:|---:|---:|---:|---:|
| black man | F4464 (+2.36 SD) | F4408 (+2.59 SD) | F4044 (+2.39 SD) | F383 (+0.90 SD) | F3460 (+2.23 SD) |
| east asian man | F1042 (+2.45 SD) | F383 (+0.14 SD) | F1130 (+2.11 SD) | F354 (+2.04 SD) | F2913 (+1.13 SD) |
| hispanic man | F76 (+2.03 SD) | F4922 (+1.81 SD) | F3936 (+1.03 SD) | F2355 (+1.32 SD) | F3351 (+1.45 SD) |
| indian man | F2784 (+2.50 SD) | F1243 (+1.84 SD) | F2913 (+1.82 SD) | F4294 (+2.26 SD) | F3301 (+2.25 SD) |
| middle eastern man | F4147 (+2.20 SD) | F1243 (+0.77 SD) | F2355 (+1.07 SD) | F2363 (+2.08 SD) | F3239 (+2.01 SD) |
| white man | F383 (+1.42 SD) | F3166 (+1.32 SD) | F2138 (+1.66 SD) | F3549 (+1.65 SD) | F2727 (+1.06 SD) |
| black woman | F4408 (+2.58 SD) | F383 (+1.97 SD) | F4464 (+2.01 SD) | F4044 (+2.23 SD) | F4951 (+2.51 SD) |
| east asian woman | F1042 (+2.49 SD) | F2138 (+2.44 SD) | F354 (+2.25 SD) | F927 (+1.50 SD) | F3397 (+1.68 SD) |
| hispanic woman | F3166 (+1.51 SD) | F3936 (+0.77 SD) | F500 (+0.70 SD) | F1243 (+0.37 SD) | F1657 (+1.38 SD) |
| indian woman | F1243 (+2.15 SD) | F3503 (+2.34 SD) | F3507 (+2.15 SD) | F4464 (+1.20 SD) | F3351 (+2.19 SD) |
| middle eastern woman | F1243 (+0.49 SD) | F1779 (+2.17 SD) | F4743 (+2.17 SD) | F3301 (+1.97 SD) | F3442 (+0.98 SD) |
| white woman | F4461 (+1.88 SD) | F2727 (+1.23 SD) | F3132 (+1.78 SD) | F3096 (+1.76 SD) | F3287 (+2.56 SD) |

## Combined JET contact sheets by rank

Each sheet has **four layer rows** (`down.2.1`, `mid.0`, `up.0.0`, `up.0.1`)
and **12 prompt-group columns**. Each cell is the selected feature's baseline
activation overlaid on its held-out seed-150 image. Color scale is normalized
per feature, so use the overlays for spatial location/area—not cross-feature
activation magnitude.

### Rank 1

![Rank 1 JET overlays](top1_features_by_layer_contact_sheet_jet.jpg)

### Rank 2

![Rank 2 JET overlays](top2_features_by_layer_contact_sheet_jet.jpg)

### Rank 3

![Rank 3 JET overlays](top3_features_by_layer_contact_sheet_jet.jpg)

### Rank 4

![Rank 4 JET overlays](../top5_feature_overlays/top4_features_by_layer_contact_sheet_jet.jpg)

### Rank 5

![Rank 5 JET overlays](../top5_feature_overlays/top5_features_by_layer_contact_sheet_jet.jpg)

## How to use the galleries

Ranks 1–3 each have individual `activation_jet.jpg`, `signed_sweep.jpg`, and
`signed_coefficient_jet.jpg` galleries. Their sweep uses multipliers −5, −3,
−2, −1, 0, 1, 2, 3, 5. `s = 1` is unchanged; `s = 0` removes the original
feature contribution; negative values reverse it. Ranks 4–5 currently have
baseline JET overlays and quantitative results, but not signed sweeps. JET
color runs blue → cyan → green → yellow → orange → red. Read `REPORT.md` for
the intervention equation and interpretation limits.
