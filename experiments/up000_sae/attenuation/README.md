# up.0.0 attenuation reports

All ten independently ranked features use the saved evaluation sweep. Each feature report covers 12 prompt groups, seeds 30–34, and multipliers −7, −5, −3, −2, −1, 0, and +1. The visual grids use seed 30 for every group so the same portraits can be compared across features.

The 3 × 3 grid follows the mid.0 example. The first tile shows the clean baseline. The signed view uses α times the original coefficient, with warm colors for positive values and cool colors for negative values. At α=0 the signed map is zero and transparent. The post-activation view shows the feature after re-encoding the edited residual. The footprint view shows the actual per-cell perturbation L2 after the generation dtype cast. Each tile is color-scaled independently. Its caption gives the full-tensor L2 for dose comparisons.

The curves show mean intended and re-encoded activation with standard errors across five seeds. All-zero overlays indicate a zero coefficient or perturbation for that panel.

| Rank | Feature report | Peak group | Example signed grid |
| ---: | --- | --- | --- |
| 1 | [1088](feature_1088/README.md) | middle eastern woman | [View](feature_1088/signed_change_heatmaps/middle_eastern_woman_seed30.jpg) |
| 2 | [4099](feature_4099/README.md) | middle eastern woman | [View](feature_4099/signed_change_heatmaps/middle_eastern_woman_seed30.jpg) |
| 3 | [55](feature_55/README.md) | east asian woman | [View](feature_55/signed_change_heatmaps/east_asian_woman_seed30.jpg) |
| 4 | [2261](feature_2261/README.md) | middle eastern man | [View](feature_2261/signed_change_heatmaps/middle_eastern_man_seed30.jpg) |
| 5 | [4303](feature_4303/README.md) | black woman | [View](feature_4303/signed_change_heatmaps/black_woman_seed30.jpg) |
| 6 | [4063](feature_4063/README.md) | white woman | [View](feature_4063/signed_change_heatmaps/white_woman_seed30.jpg) |
| 7 | [3742](feature_3742/README.md) | middle eastern man | [View](feature_3742/signed_change_heatmaps/middle_eastern_man_seed30.jpg) |
| 8 | [3103](feature_3103/README.md) | middle eastern woman | [View](feature_3103/signed_change_heatmaps/middle_eastern_woman_seed30.jpg) |
| 9 | [93](feature_93/README.md) | indian woman | [View](feature_93/signed_change_heatmaps/indian_woman_seed30.jpg) |
| 10 | [2655](feature_2655/README.md) | middle eastern woman | [View](feature_2655/signed_change_heatmaps/middle_eastern_woman_seed30.jpg) |
