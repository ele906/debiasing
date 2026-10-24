# Face-SAE group activation scan

This exploratory scan encodes the same 1,200 held-out SDXL Turbo residuals with the
saved face-region SAE (selected epoch 20). Seeds 200–249 select candidate features;
seeds 250–299 provide a held-out check. For each prompt group, it compares the
mean code inside the automatic face box with the seed-matched average for the other
five groups of the same gender. Positive delta means higher face-box activation for
that prompt group. The groups are prompt labels, not independently verified identities.

Samples with usable face boxes: 1199/1200; feature width: 5120.
The reported 95% intervals are unadjusted across 5,120 scanned features; top-ranked
features are candidates for visual review, not confirmed concepts.

## Highest Indian-woman-associated features

| Rank | Feature | Discovery delta [95% CI] | Held-out delta [95% CI] | Discovery d |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3725 | 24.0088 [23.4768, 24.5409] | 23.5749 [23.0323, 24.1175] | 12.508 |
| 2 | 3722 | 14.2126 [13.8068, 14.6184] | 14.0759 [13.6453, 14.5065] | 9.708 |
| 3 | 4858 | 8.4042 [7.9142, 8.8942] | 7.9162 [7.4709, 8.3616] | 4.754 |
| 4 | 2527 | 6.1403 [5.5032, 6.7774] | 6.3332 [5.7419, 6.9246] | 2.671 |
| 5 | 669 | 5.6241 [4.9443, 6.3039] | 5.4480 [4.8334, 6.0626] | 2.293 |
| 6 | 1664 | 5.4569 [5.0072, 5.9067] | 5.5889 [5.1774, 6.0004] | 3.363 |
| 7 | 1050 | 3.3216 [2.8982, 3.7451] | 2.8829 [2.4713, 3.2944] | 2.174 |
| 8 | 3459 | 1.8273 [1.4881, 2.1664] | 1.8264 [1.4839, 2.1689] | 1.493 |
| 9 | 1600 | 1.8120 [1.3295, 2.2944] | 1.8549 [1.3895, 2.3203] | 1.041 |
| 10 | 351 | 1.8039 [1.4391, 2.1687] | 2.0312 [1.6305, 2.4319] | 1.371 |

See `group_effects_top20.csv` for the top candidates per prompt group and
`group_effects_all_features.csv` for all tested features. Spatial overlays are still
needed to tell whether a candidate follows a forehead mark, earrings, skin texture,
lighting, pose, or another group-correlated detail.
