# Advisor-question audit

Checked the executed additive hook, stored activation arrays, generated PNGs, pixel metrics, plot preparation and plotting code. No model inference or generation was performed. Original experiment directories were left unchanged.

- Addition changes the selected SAE coordinate at all256 positions of the16×16 grid, not a face/beard mask or output RGB pixels. The model hook adds the decoder-projected difference to the existing block output.
- Jet images show the deliberately modified coordinate. Baseline tiles use the original pre-edit coordinate, verified equal across all three offsets. Addition does not raise the baseline. The broad edited tint is a plotting overlay; it is not an attention map or an alteration of the source PNG. Scales are shared within a feature across genders/strengths, not across different features.
- The saved table records4200/4200 additive edits as pixel-different from their paired baseline. Fresh checks on feature4228,+50,seed30 across all12 prompt groups confirm different PNG pixels and exact original+50 modified maps. sample_checks.json preserves those checks. Pixel difference alone does not validate semantic attribute changes.
- Multiplicative inputs use20 seeds30–49; additive uses5 seeds30–34. Photo galleries use seed30 only. Per-point averaging omits undefined beard/face ratios when predicted face area is zero, so some extreme multipliers have fewer than20 normalized scores. At α=−7,0,1,+7 the all-seed curves have20 scores per prompt group. Baseline-bearded subsets use their qualifying seeds only.
- Intervals are percentile bootstrap intervals from10000 seed resamples with RNG seed4228, not pixel-level resampling. Per-group lines summarize paired changes from the same prompt/seed control.
- The scatterplot x-axis should be the post-casting change in the actual block output: sqrt(sum((edited.float()-original.float())**2)). It is not a regularization penalty. Individual scatter points are conditions rather than seed means.

## Correction found

The saved additive generator correctly recorded both latent perturbation_norm and actual decoded_perturbation_norm. Scoring preparation mistakenly copied perturbation_norm into applied_l2 for additive rows, while multiplicative rows correctly used delta_fro_applied. This audit creates corrected derived tables and scatterplots using decoded_perturbation_norm. Across4200 edits, the largest relative difference is0.083463%; all generated images, parsed areas, paired deltas and confidence intervals are unchanged. Original tables and figures remain in their original local directories. norm_field_corrections.csv lists every correction.

The slide deck referenced by the advisor was not identified, so slide22 and slides29–41 cannot be mapped conclusively to figure filenames. In particular, a high baseline panel cannot be explained by the additive offset without inspecting that slide.
