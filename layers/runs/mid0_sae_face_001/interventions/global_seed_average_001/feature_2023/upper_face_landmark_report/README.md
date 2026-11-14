# Feature 2023 upper-face landmark analysis

Face Landmarker anchors a soft forehead/upper-face quadrilateral from the forehead landmark to the eyebrow line, spanning temple to temple. This is a geometric ROI; it does not identify hijab or other coverings.

The jet maps show the fraction of seeds whose pixel changes from baseline by RGB MAE ≥ 12.0, within that image's upper-face ROI. The cyan contour marks the mean ROI boundary. The blurred maps show mean pixel change within that ROI.

## Changed upper-face area by alpha

| Group | Alpha | Mean area % ± SD | Mean RGB MAE |
|---|---:|---:|---:|
| middle_eastern_man | -7 | 26.5% ± 11.2 | 12.28 |
| middle_eastern_man | -5 | 21.1% ± 9.8 | 10.51 |
| middle_eastern_man | -3 | 15.8% ± 8.8 | 8.61 |
| middle_eastern_man | -2 | 12.4% ± 7.6 | 7.35 |
| middle_eastern_man | -1 | 7.8% ± 5.5 | 5.47 |
| middle_eastern_man | +0 | 3.6% ± 3.2 | 3.29 |
| middle_eastern_man | +1 | 0.0% ± 0.0 | 0.00 |
| middle_eastern_man | +2 | 3.4% ± 2.9 | 3.11 |
| middle_eastern_man | +3 | 8.2% ± 5.1 | 5.53 |
| middle_eastern_man | +5 | 19.7% ± 10.3 | 10.25 |
| middle_eastern_man | +7 | 30.8% ± 14.8 | 14.48 |
| middle_eastern_woman | -7 | 25.9% ± 10.6 | 11.89 |
| middle_eastern_woman | -5 | 22.3% ± 10.3 | 10.35 |
| middle_eastern_woman | -3 | 17.9% ± 10.0 | 8.50 |
| middle_eastern_woman | -2 | 14.9% ± 8.8 | 7.15 |
| middle_eastern_woman | -1 | 10.9% ± 7.4 | 5.47 |
| middle_eastern_woman | +0 | 5.9% ± 4.8 | 3.34 |
| middle_eastern_woman | +1 | 0.0% ± 0.0 | 0.00 |
| middle_eastern_woman | +2 | 5.8% ± 4.2 | 3.26 |
| middle_eastern_woman | +3 | 11.2% ± 5.9 | 5.33 |
| middle_eastern_woman | +5 | 18.9% ± 7.9 | 8.58 |
| middle_eastern_woman | +7 | 24.6% ± 9.1 | 11.15 |
| indian_man | -7 | 21.9% ± 12.8 | 8.84 |
| indian_man | -5 | 18.3% ± 12.0 | 7.89 |
| indian_man | -3 | 13.8% ± 10.1 | 6.52 |
| indian_man | -2 | 10.6% ± 8.3 | 5.60 |
| indian_man | -1 | 6.8% ± 6.0 | 4.35 |
| indian_man | +0 | 3.0% ± 3.5 | 2.78 |
| indian_man | +1 | 0.0% ± 0.0 | 0.00 |
| indian_man | +2 | 2.9% ± 3.0 | 2.73 |
| indian_man | +3 | 6.4% ± 5.6 | 4.26 |
| indian_man | +5 | 13.2% ± 9.2 | 6.49 |
| indian_man | +7 | 18.7% ± 12.2 | 8.05 |
