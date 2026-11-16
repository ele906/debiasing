# Ten-feature additive interventions: old-photo gallery and research summary

**No new SDXL images were generated.** This analysis reuses all 4,200 completed additive edits: ten features × twelve prompt groups × five seeds × seven strengths. Original images and prior experiments are unchanged.

[Open the existing-photo gallery on GitHub](https://github.com/ele906/debiasing/tree/mid0-additive-smoke-20261114/reports/mid0_additive_smoke_20261114/ten_feature_gallery). The local gallery is in gallery/.

## Main conclusion

**The additive interventions are not literal no-ops. They change image pixels. Reliable semantic control has not been established, which is a different claim. These results do not show that race itself is the cause of the remaining appearance differences.**

Exactly pixel-identical edited images: **0/4,200**. Across the seven tested strengths, pixel MAE ranges from **0.2824 to 23.1654** on the 0–255 RGB scale; SSIM ranges from **0.6528 to 0.9975**. These quantities measure image differences, not beard presence, demographic identity or perceptual quality.

## Per-feature results

The following means use all twelve prompt groups and average within each of the five paired generation seeds. Matched-small and matched-moderate target applied block-output norms are feature-specific; they match groups within a feature, not dose across different features. Full 95% seed-bootstrap intervals and active/inactive summaries are in results/feature_strength_summary.csv.

| Feature | Initially inactive contexts /60 | Matched-small MAE | Matched-moderate MAE | Add+50 MAE | Exact pixel no-ops /420 |
|---|---:|---:|---:|---:|---:|
| 4228 | 34 | 0.701 | 0.917 | 7.135 | 0 |
| 4554 | 16 | 0.580 | 1.054 | 8.022 | 0 |
| 1650 | 34 | 0.585 | 0.653 | 6.866 | 0 |
| 4586 | 27 | 0.546 | 0.678 | 5.693 | 0 |
| 5094 | 33 | 0.578 | 0.868 | 6.468 | 0 |
| 2023 | 19 | 0.588 | 1.129 | 5.958 | 0 |
| 1074 | 31 | 0.531 | 0.746 | 5.743 | 0 |
| 4338 | 5 | 1.379 | 1.829 | 9.663 | 0 |
| 2361 | 17 | 0.786 | 1.143 | 11.067 | 0 |
| 4399 | 10 | 0.577 | 0.890 | 6.785 | 0 |

## How the interventions differ

Multiplication changes the selected coefficient by (alpha−1)z. It is a no-op when z is exactly zero. Addition changes it by beta at every spatial position and can introduce a previously inactive coordinate. The existing hook adds the corresponding decoder-direction change to the original block output, preserving the SAE reconstruction residual. Alpha=1 is therefore an exact no-op in this implementation; full SAE reconstruction is a separate control.

The seven additive levels are distribution-calibrated small and moderate offsets, two actual-norm-matched offsets, and fixed beta=10,20,50. Negative multipliers and the signed fixed-value sweep belong to earlier distinct experiments; this table does not silently combine them.

## What weak or inconsistent effects mean

The most defensible wording is: “The tested single-coordinate interventions produced measurable image changes, but the present measurements do not establish reliable, monotonic semantic control across prompt contexts.” Feature4228 should not be described as a proven beard feature.

This report does not measure beard coverage or other semantic attributes across all ten features. Nor does SSIM establish that identity, demographic appearance or image quality is preserved. No equivalence-to-zero threshold was specified, so small changes or lack of an obvious thumbnail difference cannot establish a formal null effect.

## Why this does not establish race as the cause

Prompt conditioning, other SAE coordinates, their interactions, spatial alignment, the rest of the diffusion representation, intervention timing, and evaluator sensitivity remain possible explanations. Race words in a prompt are conditioning variables; they are not an experimentally isolated biological cause.

A weak response to adding one coordinate does not isolate the mechanism that creates the original appearance. The coordinate may be insufficient, redundant, context-dependent, incorrectly interpreted, or edited in an unnatural spatial configuration. These alternatives are not distinguished by the present data. Neither “activations do nothing” nor “it must be race itself” follows.

## Scope and uncertainty

There are five paired generation seeds, not4,200 independent experimental units. Confidence intervals resample complete seed clusters while retaining all prompt groups. Active/inactive contexts are observational strata: their prompt-group composition differs, so their mean difference is not a randomized causal effect of activity. Dose comparisons across features must account for the recorded applied norms.

Checkpoint: layers/runs/mid0_sae_face_001/sae; SHA-256 cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6. Only unet.mid_block.attentions.0 and the same frozen single-step SDXL Turbo generator are represented.

## Files and reproduction

gallery/: derived contact sheets from saved images. results/gallery_sources.csv: source paths and hashes for every gallery cell. results/per_image_pixel_results.csv: all4,200 evaluated originals and paired baselines. results/feature_strength_summary.csv: seed-cluster intervals, active/inactive summaries and actual perturbation norms. report.tex and report.pdf: LaTeX research summary. code/: reproducible gallery and resumable image-metric scripts.

Run pixel_metrics.py --run THIS_DIRECTORY in the recorded existing analysis environment to resume missing image evaluations; no model is loaded. The image cache is append-only. All originals and activation archives remain in the earlier generation directories.
