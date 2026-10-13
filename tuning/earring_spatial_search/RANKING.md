# Spatial SAE feature search for earrings

For each same-prompt East-Asian-woman image, we encoded the saved UNet activation with the pretrained SAE. We scored each feature by its mean activation inside the C2C earring box (positive images) or canonical earring regions derived from training positives (negative images). Candidate ranking uses train images only; validation results are held out.

Train: 60 images (19 detector-positive, 41 detector-negative). Validation: 20 (8 positive, 12 negative). Labels are automatic and unreviewed.

| Rank | Feature | Train AUC | Validation AUC | Train positive images active | Validation positive images active |
|---:|---:|---:|---:|---:|---:|
| 1 | 4858 | 0.670 | 0.448 | 19 | 8 |
| 2 | 766 | 0.658 | 0.562 | 6 | 1 |
| 3 | 1237 | 0.603 | 0.500 | 8 | 2 |
| 4 | 2137 | 0.603 | 0.417 | 13 | 5 |
| 5 | 527 | 0.599 | 0.526 | 8 | 1 |
| 6 | 2940 | 0.598 | 0.500 | 15 | 7 |
| 7 | 240 | 0.594 | 0.490 | 12 | 4 |
| 8 | 1663 | 0.582 | 0.500 | 4 | 0 |
| 9 | 2233 | 0.567 | 0.458 | 3 | 0 |
| 10 | 1714 | 0.556 | 0.526 | 3 | 1 |
| 11 | 2230 | 0.556 | 0.583 | 18 | 8 |
| 12 | 310 | 0.553 | 0.562 | 2 | 1 |
| 13 | 1953 | 0.553 | 0.458 | 2 | 0 |
| 14 | 2430 | 0.553 | 0.625 | 2 | 2 |
| 15 | 2874 | 0.553 | 0.562 | 2 | 1 |
| 16 | 3029 | 0.553 | 0.500 | 2 | 0 |
| 17 | 3749 | 0.553 | 0.562 | 2 | 1 |
| 18 | 546 | 0.551 | 0.469 | 16 | 8 |
| 19 | 3267 | 0.549 | 0.479 | 6 | 2 |
| 20 | 1855 | 0.548 | 0.594 | 3 | 2 |
| 21 | 433 | 0.546 | 0.562 | 3 | 1 |
| 22 | 4254 | 0.542 | 0.458 | 2 | 0 |
| 23 | 3464 | 0.540 | 0.562 | 2 | 1 |
| 24 | 243 | 0.537 | 0.422 | 5 | 3 |
| 25 | 1563 | 0.530 | 0.427 | 16 | 4 |

AUC 0.5 is chance ordering; values near 1 indicate stronger separation. Because C2C labels are noisy and this is one prompt group, high validation AUC is only a candidate signal. Next, the top three train-ranked features are tested with spatial box masks on validation positives against feature 2940 as a control. Visual review is still required.
