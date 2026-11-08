# Feature 2023 landmark-anchored head-covering-region analysis

Face Landmarker anchors a soft region that extends above the forehead and beyond the temples, down to the eyebrow line. It includes surrounding head context where a covering may sit. This is a geometric ROI, not hijab or semantic covering detection.

Jet maps show the fraction of seeds whose pixel changes from baseline by RGB MAE ≥ 12.0, within each image's landmark-anchored region. The cyan contour marks the mean region boundary. Blurred maps show mean pixel change inside the same region.

## Changed area by alpha

| Group | Alpha | Mean area % ± SD | Mean RGB MAE |
|---|---:|---:|---:|
| middle_eastern_man | -7 | 53.7% ± 15.3 | 33.44 |
| middle_eastern_man | -5 | 47.9% ± 14.7 | 29.07 |
| middle_eastern_man | -3 | 41.0% ± 14.4 | 23.53 |
| middle_eastern_man | -2 | 35.8% ± 14.2 | 19.57 |
| middle_eastern_man | -1 | 28.2% ± 13.1 | 14.72 |
| middle_eastern_man | +0 | 16.4% ± 9.5 | 8.63 |
| middle_eastern_man | +1 | 0.0% ± 0.0 | 0.00 |
| middle_eastern_man | +2 | 17.7% ± 11.1 | 9.31 |
| middle_eastern_man | +3 | 29.8% ± 14.3 | 17.00 |
| middle_eastern_man | +5 | 46.7% ± 15.8 | 27.53 |
| middle_eastern_man | +7 | 56.3% ± 15.6 | 36.64 |
| middle_eastern_woman | -7 | 44.4% ± 11.5 | 19.63 |
| middle_eastern_woman | -5 | 39.4% ± 11.4 | 16.96 |
| middle_eastern_woman | -3 | 33.0% ± 10.8 | 13.82 |
| middle_eastern_woman | -2 | 28.4% ± 10.0 | 11.73 |
| middle_eastern_woman | -1 | 21.9% ± 8.5 | 9.11 |
| middle_eastern_woman | +0 | 12.6% ± 5.7 | 5.73 |
| middle_eastern_woman | +1 | 0.0% ± 0.0 | 0.00 |
| middle_eastern_woman | +2 | 12.2% ± 4.6 | 5.57 |
| middle_eastern_woman | +3 | 21.2% ± 5.6 | 8.62 |
| middle_eastern_woman | +5 | 32.6% ± 6.2 | 12.94 |
| middle_eastern_woman | +7 | 40.3% ± 6.5 | 16.26 |
| indian_man | -7 | 41.9% ± 14.1 | 15.40 |
| indian_man | -5 | 35.4% ± 12.6 | 12.93 |
| indian_man | -3 | 26.8% ± 10.7 | 10.23 |
| indian_man | -2 | 21.2% ± 9.2 | 8.57 |
| indian_man | -1 | 14.4% ± 7.1 | 6.53 |
| indian_man | +0 | 6.3% ± 4.2 | 3.97 |
| indian_man | +1 | 0.0% ± 0.0 | 0.00 |
| indian_man | +2 | 6.1% ± 4.0 | 3.89 |
| indian_man | +3 | 13.4% ± 7.0 | 6.26 |
| indian_man | +5 | 25.1% ± 9.9 | 9.84 |
| indian_man | +7 | 34.3% ± 11.6 | 12.91 |
