# Activation tuning

This folder contains feature intervention experiments on images from the original biased race/gender dataset. Each feature gets its own folder with the sweep script, notebook, run metadata, metrics, charts, and image sheets.

## Feature 138

The first run uses the 100 Middle Eastern women images because feature 138 had its largest mean activation in that group. Prompts and seeds come from `backprop/manifest.csv` and use the original template, `a picture of a {race} {gender}`. The intervention varies alpha across `-100, -80, -60, -50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50, 60, 80, 100`; each seed sheet contains the original and all 19 alpha variants, ordered ascending.

The earlier scarf/no-scarf prompt experiment remains in `../race_gender_feature_interventions/feature_138/` as a separate prompt-controlled exploration. It is not used for the biased-dataset results here.

## Feature 3884

The follow-up run covers all 12 biased race/gender groups, with 20 original seeds per group. The symmetric alpha set is `-75, -50, -30, -20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20, 30, 50, 75`. Browse the results in [feature 3884](feature_3884/README.md), organized by group with a 5×4 grid per seed.

The runnable notebook and reusable scripts are also included in the public [cos398 `tuning_activations` folder](https://github.com/ele906/cos398/tree/main/tuning_activations) and the [SAE demo notebook repository](https://github.com/ele906/sae-demo-notebooks/tree/main/sae-activation/activation_tuning).

## Candidate multiplier histograms

Each completed feature sweep has a histogram preview in its feature folder. The
plots show the visually selected candidate multipliers listed in
[`REASONABLE_MULTIPLIERS.md`](REASONABLE_MULTIPLIERS.md). Feature 3611 has no
candidate interval in that review, so its plot shows the full saved multiplier
sweep and should be read as a diagnostic preview, not a recommended range.

The upper panel stacks the number of images with a nonzero scaled feature by
group and multiplier. The lower panel shows the distribution of signed scaled
coefficients at positions active before intervention, normalized as a
percentage; multiplier 0 is omitted from this panel because it produces only
zero coefficients. It uses the same fixed −550 to +550 coefficient axis as the
feature 138 reference plot, so values outside that range are clipped from view.
These plots summarize activation-map behavior; they do not validate image
quality, safety bounds, or universal clamps.

To regenerate all plots from the saved maps and this folder's candidate table,
run:

```bash
python render_candidate_histograms.py
```

| Feature | Plot |
|---:|---|
| 138 | [histogram](feature_138/feature_138_candidate_histogram.png) |
| 1563 | [histogram](feature_1563/feature_1563_candidate_histogram.png) |
| 1664 | [histogram](feature_1664/feature_1664_candidate_histogram.png) |
| 1759 | [histogram](feature_1759/feature_1759_candidate_histogram.png) |
| 199 | [histogram](feature_199/feature_199_candidate_histogram.png) |
| 2486 | [histogram](feature_2486/feature_2486_candidate_histogram.png) |
| 2527 | [histogram](feature_2527/feature_2527_candidate_histogram.png) |
| 3181 | [histogram](feature_3181/feature_3181_candidate_histogram.png) |
| 3279 | [histogram](feature_3279/feature_3279_candidate_histogram.png) |
| 3459 | [histogram](feature_3459/feature_3459_candidate_histogram.png) |
| 3611 | [histogram](feature_3611/feature_3611_candidate_histogram.png) |
| 3722 | [histogram](feature_3722/feature_3722_candidate_histogram.png) |
| 3725 | [histogram](feature_3725/feature_3725_candidate_histogram.png) |
| 3884 | [histogram](feature_3884/feature_3884_candidate_histogram.png) |
| 433 | [histogram](feature_433/feature_433_candidate_histogram.png) |
| 452 | [histogram](feature_452/feature_452_candidate_histogram.png) |
| 527 | [histogram](feature_527/feature_527_candidate_histogram.png) |
| 546 | [histogram](feature_546/feature_546_candidate_histogram.png) |
| 669 | [histogram](feature_669/feature_669_candidate_histogram.png) |
