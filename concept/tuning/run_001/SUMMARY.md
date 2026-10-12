# Tuning sweep summary

Detector presence is an unreviewed proxy. Lost labels do not establish concept removal or reduced racism. Requested present/absent conditions can disagree with what was actually generated.

| Role | Feature | Alpha | Images | Baseline labels | Edited labels | Labels lost | Labels gained | Mean pixel change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target | 3181 | 0 | 16 | 9 | 9 | 0 | 0 | 6.021 |
| target | 3181 | 0.5 | 16 | 9 | 9 | 0 | 0 | 3.840 |
| target | 3181 | 1 | 16 | 9 | 9 | 0 | 0 | 0.000 |
| target | 3181 | 1.5 | 16 | 9 | 9 | 0 | 0 | 3.828 |
| control | 2808 | 0 | 16 | 9 | 10 | 0 | 1 | 6.814 |
| control | 2808 | 0.5 | 16 | 9 | 9 | 0 | 0 | 4.033 |
| control | 2808 | 1 | 16 | 9 | 9 | 0 | 0 | 0.000 |
| control | 2808 | 1.5 | 16 | 9 | 9 | 0 | 0 | 4.222 |

Pixel MAE is on RGB values from 0 to 255. Region measurements use the baseline detector boxes. The control is matched approximately on activity; it is not known to be semantically neutral. Inspect target/control grids and review.csv for necklace change, face/clothing damage, artifacts, and prompt fidelity. No best alpha has been selected.

[All comparison grids](REPORT.md)
[Full measurements](metrics.csv)
[Blank review sheet](review.csv)
