# Feature 4228 beard-mask feasibility probe

## What was tried

SCHP ATR predicts a generic `hair` class. I intersected that mask with the
existing Face-Landmarker jaw/chin ROI to make a candidate beard mask, then
reviewed six Indian-man samples (seeds 30 and 31; alphas -7, +1, +7).

## Initial result

The candidate does **not** reliably trace the visible beard. In the reviewed
overlays, the model often misses the white facial hair and marks small cheek or
side-of-face regions instead. Treat the pixel counts as a failed feasibility
probe, not beard-area measurements. Do not compare these values across alpha.

| Seed | Alpha | Candidate pixels / lower-face ROI | Candidate share |
|---:|---:|---:|---:|
| 30 | -7 | 372 / 59,845 | 0.62% |
| 30 | +1 | 621 / 59,009 | 1.05% |
| 30 | +7 | 242 / 58,032 | 0.42% |
| 31 | -7 | 17 / 49,856 | 0.03% |
| 31 | +1 | 574 / 50,173 | 1.14% |
| 31 | +7 | 1,902 / 48,776 | 3.90% |

The variation is consistent with poor beard coverage and/or unstable class
assignment, so these numbers should not be interpreted as beard growth or loss.

## Visual review

- [Seed 30, alpha -7](indian_man/seed030_alpha_m7_candidate_overlay.png)
- [Seed 30, alpha +1](indian_man/seed030_alpha_p1_candidate_overlay.png)
- [Seed 30, alpha +7](indian_man/seed030_alpha_p7_candidate_overlay.png)
- [Seed 31, alpha -7](indian_man/seed031_alpha_m7_candidate_overlay.png)
- [Seed 31, alpha +1](indian_man/seed031_alpha_p1_candidate_overlay.png)
- [Seed 31, alpha +7](indian_man/seed031_alpha_p7_candidate_overlay.png)
- [Raw pixel counts](probe_metrics.csv)

## Next useful experiment

Use manual beard masks on a small, blinded sample to establish what the beard
boundary should look like. If an automatic method is needed, evaluate a
facial-hair-specific segmentation model against those masks first. The ATR
human-parsing label set contains `hair` and `face`, but no beard category, so
its generic hair prediction is not a validated beard segmenter.
