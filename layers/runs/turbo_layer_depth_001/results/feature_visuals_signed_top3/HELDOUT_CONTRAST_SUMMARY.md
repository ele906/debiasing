# Held-out feature-contrast summary

This is the effect-size companion to the [spatial activation-area summary](SPATIAL_ACTIVATION_SUMMARY.md).
It asks whether a discovery-selected feature remains more active for its target
prompt condition on different, held-out seeds.

## Measurement

- **Discovery:** feature IDs and ranks were selected only on seeds 100–149.
- **Evaluation:** seeds 150–199, paired within seed and gender.
- **Per group contrast:** target-prompt activation minus the mean activation of
  the other five race-prompt conditions with the same gender.
- **Standardization:** divide that held-out contrast by the held-out standard
  deviation of the selected feature over same-gender prompt samples.

Thus, `+2.13 SD` means that a selected feature was, on average, 2.13 of its
own held-out activation standard deviations above the matched reference prompts.
This is a feature-activation contrast—not a semantic label, fairness metric, or
image-quality metric.

## Mean held-out contrast, in feature SD

Each cell averages 12 prompt groups after the rank was frozen on discovery data.

| Layer | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| down.2.1 | +2.13 SD | +2.04 SD | +1.69 SD |
| mid.0 | +1.23 SD | +1.18 SD | +1.02 SD |
| up.0.0 | +1.41 SD | +1.42 SD | +1.17 SD |
| up.0.1 | +2.00 SD | +1.61 SD | +1.76 SD |

## Main observation

The strongest mean held-out Rank-1 contrast is down.2.1 (+2.13 SD), followed
closely by up.0.1 (+2.00 SD). Mid.0 is lower (+1.23 SD). This pattern does not
establish a simple monotonic down-to-up trend, but it shows that the
discovery-selected features retain sizable prompt-condition contrasts on held-out
seeds.

The full per-group feature IDs and effects are in
[TOP_FEATURES_BY_LAYER.md](TOP_FEATURES_BY_LAYER.md).
