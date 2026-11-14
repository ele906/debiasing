# Activation tuning

This folder contains feature intervention experiments on images from the original biased race/gender dataset. Each feature gets its own folder with the sweep script, notebook, run metadata, metrics, charts, and image sheets.

## Feature 138

`run_biased_001` is the original focused run: it uses 100 Middle Eastern women images because feature 138 had its largest mean activation in that group. `run_biased_002` is the comparable all-group run: it uses the first 20 seeds for each of the 12 race/gender groups, matching the sample count used by the other feature folders. Prompts and seeds come from `backprop/manifest.csv` and use the original template, `a picture of a {race} {gender}`. The intervention varies alpha across `-100, -80, -60, -50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50, 60, 80, 100`.

Run the full comparison with `python sae-activation-tuning/tuning_activations/feature_138/run_biased_dataset_sweep.py`. The default writes to `run_biased_002` and refuses to overwrite an existing output directory. Set `SAE_FEATURE_RUN` to that output path and run `render_signed_visuals.py` to render the sheets and overlays.

The earlier scarf/no-scarf prompt experiment remains in `../race_gender_feature_interventions/feature_138/` as a separate prompt-controlled exploration. It is not used for the biased-dataset results here.

`feature_138_histogram_preview_stacked.png` is rendered from the saved `run_biased_002/activation_maps.npz` by `render_histogram_preview.py`. Its top panel reports raw image counts: an image counts as active when at least one spatial position has a nonzero signed scaled coefficient, including negative coefficients. The bottom panel shows the signed scaled-coefficient distribution. Both panels use symmetric x limits so zero aligns vertically.

The run folders contain the per-image generated outputs locally. The GitHub result package contains the aggregate metrics, activation maps, settings, and visual-review sheets; the generated image folders are multi-gigabyte.

The runnable notebook and reusable scripts are also included in the public [cos398 `tuning_activations` folder](https://github.com/ele906/cos398/tree/main/tuning_activations) and the [SAE demo notebook repository](https://github.com/ele906/sae-demo-notebooks/tree/main/sae-activation/activation_tuning).
