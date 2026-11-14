# Spatial-pattern experiment: exploratory results

220/220 planned conditions complete; 200 edited conditions at matched targets 42.5242 and 65.1363. Maximum relative norm mismatch: 6.269121e-06. Two frozen donor maps, three shuffles each, and five paired seeds.

All no-op replays were checked against saved baseline pixels. Feature activation was checked per spatial map: East-Asian-man seed 33 has two active cells, while seeds 30,31,32,34 are exactly inactive. Black-man seeds 32 and 34 are also inactive.

0 blinded human-rating observations currently available. Facial-hair presence, coverage and structural-validity effects remain pending when corresponding ratings are absent. The facial-hair figure explicitly displays this missing measurement. Pixel differences, SSIM, face detection and demographic classifier output are not substitutes for beard judgments.

Paired comparisons average donor and shuffle replicates inside each seed, then compare natural versus uniform, shuffled and localized patterns. Intervals use the exact empirical seed-cluster bootstrap (all 5^5 resamples for n=5). Two-sided sign-flip p-values enumerate all 2^5 sign assignments. All analyses are exploratory, with no multiple-comparison-adjusted confirmatory claim. Intervals can be degenerate at boundaries; five seeds provide limited population inference. Donor maps are fixed, not a random sample of features or donors.

Naturalness is measured separately: support expansion beyond k=10, effective spatial positions, peak coordinate changes, lower-face pattern energy and post-reencoding discrepancy. Equal output L2 norms do not equate these properties. The localized control is the recipient's geometric lower-face mask; raw donor maps are transferred without pose alignment or warping.

See paired_comparisons.csv for per-donor and pooled estimates, per_image_results.csv for every evaluated image, and failures_and_missingness.csv for generation failures. Per-image classifier confidence vectors and landmarks remain in evaluations/. No demographic causal specificity or beard-control claim follows from these proxy metrics alone.

## Observed proxy results at L2 65.1363

| Context | Method | Pixel MAE | Lower-face fraction of change | Cells beyond k=10 |
|---|---|---:|---:|---:|
| black_man | natural | 0.537 | 0.292 | 4.1% |
| black_man | uniform | 0.684 | 0.253 | 96.6% |
| black_man | shuffled | 0.466 | 0.285 | 5.8% |
| black_man | localized | 0.854 | 0.320 | 28.6% |
| east_asian_man | natural | 0.629 | 0.252 | 6.0% |
| east_asian_man | uniform | 0.777 | 0.231 | 99.8% |
| east_asian_man | shuffled | 0.549 | 0.245 | 6.1% |
| east_asian_man | localized | 1.412 | 0.278 | 22.0% |

For East-Asian-man prompts, the donor-averaged natural-minus-shuffled lower-face change fraction at L2=65.1363 is 0.00784 (95% seed-bootstrap interval −0.01036 to 0.02715; exact two-sided sign-flip p=0.5). This small experiment does not establish a clear localization advantage of the natural layout over shuffled layouts. It provides no validated semantic beard effect yet.

All 220 images were detected by MediaPipe and received FairFace probabilities. Structural validity remains unreviewed. Wilson intervals in the main detection figure treat each seed as a success only if all method replicates were detected; with 5/5 successes the 95% lower bound is approximately 0.566. The original empirical-bootstrap detection figure is retained under a distinct filename because boundary bootstrap intervals are degenerate.
