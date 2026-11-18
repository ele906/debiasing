# SAE feature prevalence and JET overlays

The ten features selected at up.0.0 are the ten highest-disparity features across all 5,120 directions. Disparity is the maximum prompt-group mean activation minus the mean across the other eleven groups. Prevalence is the fraction of positive spatial tokens. It measures how often a feature activates, while disparity also reflects activation magnitude. Feature IDs are specific to their layer.

Each prevalence plot shows all 12 prompt groups and the per-image histogram for the peak-mean group versus the other eleven groups. Histograms are normalized within each subset and retain the zero-activation bin. The JET sheets use seeds 185, 192, and 199 across every group, with one shared color scale per feature.

[Group prevalence table](feature_prevalence/group_prevalence.csv) · [JET scales and seeds](multiseed_jet/settings.json)

[Mean-activation bar charts in the mid.0 reference format](GROUP_ACTIVATION_CHARTS.md)

[Per-feature attenuation grids and curves](attenuation/README.md)

## up.0.0

| Rank | Feature | Peak-mean prompt group | Disparity | Prevalence | JET overlays |
| ---: | ---: | --- | ---: | --- | --- |
| 1 | 1088 | middle eastern woman | 18.51 | [Histogram](feature_prevalence/up.0.0_feature_1088.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_1088.jpg) |
| 2 | 4099 | middle eastern woman | 17.87 | [Histogram](feature_prevalence/up.0.0_feature_4099.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_4099.jpg) |
| 3 | 55 | east asian woman | 13.08 | [Histogram](feature_prevalence/up.0.0_feature_55.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_55.jpg) |
| 4 | 2261 | middle eastern man | 12.90 | [Histogram](feature_prevalence/up.0.0_feature_2261.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_2261.jpg) |
| 5 | 4303 | black woman | 10.27 | [Histogram](feature_prevalence/up.0.0_feature_4303.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_4303.jpg) |
| 6 | 4063 | white woman | 8.52 | [Histogram](feature_prevalence/up.0.0_feature_4063.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_4063.jpg) |
| 7 | 3742 | middle eastern man | 7.79 | [Histogram](feature_prevalence/up.0.0_feature_3742.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_3742.jpg) |
| 8 | 3103 | middle eastern woman | 7.35 | [Histogram](feature_prevalence/up.0.0_feature_3103.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_3103.jpg) |
| 9 | 93 | indian woman | 7.29 | [Histogram](feature_prevalence/up.0.0_feature_93.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_93.jpg) |
| 10 | 2655 | middle eastern woman | 7.24 | [Histogram](feature_prevalence/up.0.0_feature_2655.png) | [3-seed sheet](multiseed_jet/up.0.0_feature_2655.jpg) |

## mid.0

| Rank | Feature | Peak-mean prompt group | Disparity | Prevalence | JET overlays |
| ---: | ---: | --- | ---: | --- | --- |
| 1 | 4228 | middle eastern man | 5.88 | [Histogram](feature_prevalence/mid.0_feature_4228.png) | [3-seed sheet](multiseed_jet/mid.0_feature_4228.jpg) |
| 2 | 4554 | white woman | 4.07 | [Histogram](feature_prevalence/mid.0_feature_4554.png) | [3-seed sheet](multiseed_jet/mid.0_feature_4554.jpg) |
| 3 | 1650 | black woman | 3.93 | [Histogram](feature_prevalence/mid.0_feature_1650.png) | [3-seed sheet](multiseed_jet/mid.0_feature_1650.jpg) |
| 4 | 4586 | white woman | 3.56 | [Histogram](feature_prevalence/mid.0_feature_4586.png) | [3-seed sheet](multiseed_jet/mid.0_feature_4586.jpg) |
| 5 | 5094 | white woman | 3.37 | [Histogram](feature_prevalence/mid.0_feature_5094.png) | [3-seed sheet](multiseed_jet/mid.0_feature_5094.jpg) |
| 6 | 2023 | middle eastern man | 3.23 | [Histogram](feature_prevalence/mid.0_feature_2023.png) | [3-seed sheet](multiseed_jet/mid.0_feature_2023.jpg) |
| 7 | 1074 | white woman | 3.09 | [Histogram](feature_prevalence/mid.0_feature_1074.png) | [3-seed sheet](multiseed_jet/mid.0_feature_1074.jpg) |
| 8 | 4338 | white man | 2.86 | [Histogram](feature_prevalence/mid.0_feature_4338.png) | [3-seed sheet](multiseed_jet/mid.0_feature_4338.jpg) |
| 9 | 2361 | white woman | 2.45 | [Histogram](feature_prevalence/mid.0_feature_2361.png) | [3-seed sheet](multiseed_jet/mid.0_feature_2361.jpg) |
| 10 | 4399 | middle eastern woman | 2.31 | [Histogram](feature_prevalence/mid.0_feature_4399.png) | [3-seed sheet](multiseed_jet/mid.0_feature_4399.jpg) |
