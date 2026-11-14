# Balanced prompt-group comparison

This reuses the existing 1,200-image face-SAE group scan. Discovery seeds 200–249 selected candidate features; held-out seeds 250–299 provide the comparison shown here. For each race/gender prompt cell, the contrast is against the seed-matched other five race prompts of the same gender. It balances seed and gender comparisons, but the groups remain generated prompt conditions rather than verified identities.

The heatmap shows held-out contrasts for eight candidates: 3725 and 3722 (Indian-woman-associated features), 4858 (ear-region candidate), 3593/3701 (eye candidates), 1600 (eyebrows), 1664 (nose), and 527 (group-associated high activation). These IDs are fixed candidates from prior reports, not features selected on this held-out table. Positive cells indicate higher mean activation relative to the same-gender reference groups.

This remains an association scan. It cannot separate facial appearance from head covering, clothing, lighting, pose, or other prompt-correlated details. See `selected_feature_group_effects.csv` for estimates and unadjusted intervals.
