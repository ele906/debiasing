# Overnight face-feature screen

This automated screen compares six candidate face-localized SAE features on the biased/simple-prompt dataset. It uses held-out seeds 20–29 across all 12 prompt categories. Each run tests alpha 0 (remove), +1 (control), and +3 (boost).

The men/women pages show **original | +3 boost | amplified pixel difference**. For readability, each group displays the held-out seed with the largest +3 pixel change; these are screening visualizations, not average-effect claims.

| Feature | All men | All women | Settings | Mean metrics |
|---:|---|---|---|---|
| 1874 | [comparison](feature_1874/all_men.png) | [comparison](feature_1874/all_women.png) | [settings](feature_1874/settings.json) | [metrics](feature_1874/mean_metrics_by_group_alpha.csv) |
| 4228 | [comparison](feature_4228/all_men.png) | [comparison](feature_4228/all_women.png) | [settings](feature_4228/settings.json) | [metrics](feature_4228/mean_metrics_by_group_alpha.csv) |
| 4338 | [comparison](feature_4338/all_men.png) | [comparison](feature_4338/all_women.png) | [settings](feature_4338/settings.json) | [metrics](feature_4338/mean_metrics_by_group_alpha.csv) |
| 3134 | [comparison](feature_3134/all_men.png) | [comparison](feature_3134/all_women.png) | [settings](feature_3134/settings.json) | [metrics](feature_3134/mean_metrics_by_group_alpha.csv) |
| 4601 | [comparison](feature_4601/all_men.png) | [comparison](feature_4601/all_women.png) | [settings](feature_4601/settings.json) | [metrics](feature_4601/mean_metrics_by_group_alpha.csv) |
| 4305 | [comparison](feature_4305/all_men.png) | [comparison](feature_4305/all_women.png) | [settings](feature_4305/settings.json) | [metrics](feature_4305/mean_metrics_by_group_alpha.csv) |

## Interpretation

A useful feature should make a visible, repeatable, spatially localized edit while preserving identity and image realism. The screen is used to choose candidates for a later small spatial bundle; it does not assign semantic labels automatically.
