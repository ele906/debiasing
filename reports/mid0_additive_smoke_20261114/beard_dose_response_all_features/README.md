# Existing-image beard-parser intervention scoring

{
  "n_conditions": 74400,
  "n_unique_assets": 49520,
  "n_new_generated_images": 0,
  "n_trained_models": 0,
  "features": [
    1074,
    1650,
    2023,
    2361,
    4228,
    4338,
    4399,
    4554,
    4586,
    5094
  ],
  "n_private_review_joins": 100,
  "bootstrap_seed": 4228,
  "n_bootstrap": 10000,
  "random_control_status": "No matched-norm random controls found in inspected intervention archives; other nine selected features are not random controls"
}

## Definitions and limits

The evaluator is the existing FaceParsing decoder and DINOv3 backbone. Raw class14 is listed as Beard in its README; a conflicting visualization class list remains documented. These are unvalidated parser predictions. The original helper, weights and preprocessing are unchanged.

Primary area fraction is beard / parser classes1–11 plus14 (facial components and beard; excludes neck12 and hair13). Secondary metrics divide by the same-seed baseline face area held fixed, and by full512×512 image area. Current-face normalization can itself change under edits; fixed-denominator sensitivity is provided. Zero denominators stay missing. Fixed-baseline fractions can exceed100%; they are not clipped.

Every edit is paired by feature, prompt and seed with its own control. Multiplicative controls use α=1; additive uses the saved no-intervention image. config/control_equivalence.csv records actual RGB equivalence. α=0 removes the coordinate, α=1 leaves it unchanged, α<0 reverses sign, α>1 amplifies. There are no fractional α between0 and1: no graded attenuation claim is supported within that interval. Applied L2 is recorded separately; negative multipliers are not relabeled attenuation.

Full20-seed multiplicative curves and five-seed additive curves have different precision. Use results/common_seed_comparison.csv for shared seeds30–34. Per-context paired curves have one line per seed; bootstrap resamples seeds, averaging contexts within a seed before pooled summaries. Confidence intervals do not account for parser error. No multiple-testing-adjusted significance or demographic specificity is claimed. Prompt groups are conditioning text.

Baseline-bearded panels use a provisional baseline beard/face fraction≥1%; thresholds0,0.5,1,2,5% are retained for sensitivity. This is an operational parser criterion, not validated beard presence. All zero-baseline contexts remain in the full table and initiation analysis. Additive initiation is a threshold crossing, not proof of new facial hair.

Shading |α|≥50 is a caution range motivated by an existing stratified100-case human review; it is not a universal distortion threshold. MediaPipe detection is not structural validity. Per-case review linkage, where available, stays local/private; unreviewed images are not presumed usable.

## Measured results for feature4228

These are paired changes in the new parser’s predicted beard fraction of its facial classes, in percentage points. The multiplicative rows use20 seeds and the additive rows use5; their perturbation magnitudes are not matched by this table. Intervals resample seeds and do not include parser error.

| Prompt group | Intervention | Seeds | Mean change (pp) | Exploratory95% interval |
|---|---|---:|---:|---|
| indian_man | alpha_-7 | 20 | -6.42 | [-9.27, -4.01] |
| indian_man | alpha_0 | 20 | -1.62 | [-2.17, -1.07] |
| indian_man | alpha_7 | 20 | +11.18 | [+8.15, +14.32] |
| indian_man | fixed_50 | 5 | +2.32 | [+1.53, +2.99] |
| middle_eastern_man | alpha_-7 | 20 | -7.89 | [-9.15, -6.63] |
| middle_eastern_man | alpha_0 | 20 | -1.79 | [-2.14, -1.48] |
| middle_eastern_man | alpha_7 | 20 | +12.42 | [+10.83, +14.04] |
| middle_eastern_man | fixed_50 | 5 | +3.29 | [+2.44, +4.14] |
| east_asian_man | alpha_-7 | 20 | +0.05 | [-0.03, +0.16] |
| east_asian_man | alpha_0 | 20 | +0.03 | [-0.01, +0.10] |
| east_asian_man | alpha_7 | 20 | -0.01 | [-0.12, +0.07] |
| east_asian_man | fixed_50 | 5 | -0.54 | [-1.31, +0.00] |
| black_man | alpha_-7 | 20 | -0.31 | [-0.66, -0.00] |
| black_man | alpha_0 | 20 | -0.18 | [-0.31, -0.06] |
| black_man | alpha_7 | 20 | +0.51 | [+0.07, +1.02] |
| black_man | fixed_50 | 5 | +0.59 | [+0.11, +1.14] |
| hispanic_man | alpha_-7 | 20 | -0.20 | [-0.66, +0.40] |
| hispanic_man | alpha_0 | 20 | -0.04 | [-0.11, +0.04] |
| hispanic_man | alpha_7 | 20 | +0.54 | [+0.07, +1.11] |
| hispanic_man | fixed_50 | 5 | +0.76 | [-0.03, +1.57] |
| white_man | alpha_-7 | 20 | -0.12 | [-0.30, +0.00] |
| white_man | alpha_0 | 20 | -0.01 | [-0.03, +0.02] |
| white_man | alpha_7 | 20 | +0.54 | [+0.05, +1.23] |
| white_man | fixed_50 | 5 | +0.36 | [-0.04, +1.12] |

The parser responds to both multiplicative and additive interventions in some contexts. Therefore “additive features do nothing” is not supported by these measurements. Removing feature4228 (α=0) and reversing its sign (α<0) are distinct interventions. Low response in another prompt group does not establish race as a cause: natural activation, perturbation size, baseline facial hair, other representations and parser behavior remain possible explanations.

For +50 addition to feature4228, using a provisional1% beard/face threshold, parser crossings among baseline-below-threshold contexts are: black_man 0/3, black_woman 0/5, east_asian_man 0/2, east_asian_woman 0/5, hispanic_man 0/0, hispanic_woman 0/5, indian_man 0/0, indian_woman 0/5, middle_eastern_man 0/0, middle_eastern_woman 0/5, white_man 1/5, white_woman 0/5. A zero denominator here means no eligible seed contexts, not a negative result. Crossings are candidate cases for visual inspection, not validated new beards. Threshold sensitivity is retained in the CSVs.

Matched-norm random-feature/direction controls were not found in the inspected face-intervention archives. Other selected features are descriptive comparisons, not random controls. No feature-specific or demographic causal conclusion is established.

## Figures

![Ten features, multipliers −7 through +7](figures/ten_feature_multiplicative_low_range.png)

![feature1074_men_all_delta_current_face_pp](figures/feature1074_men_all_delta_current_face_pp.png)

![feature1074_men_baseline_bearded_delta_current_face_pp](figures/feature1074_men_baseline_bearded_delta_current_face_pp.png)

![feature1074_additive_fixed](figures/feature1074_additive_fixed.png)

![feature1074_applied_norm](figures/feature1074_applied_norm.png)

![feature1650_men_all_delta_current_face_pp](figures/feature1650_men_all_delta_current_face_pp.png)

![feature1650_men_baseline_bearded_delta_current_face_pp](figures/feature1650_men_baseline_bearded_delta_current_face_pp.png)

![feature1650_additive_fixed](figures/feature1650_additive_fixed.png)

![feature1650_applied_norm](figures/feature1650_applied_norm.png)

![feature2023_men_all_delta_current_face_pp](figures/feature2023_men_all_delta_current_face_pp.png)

![feature2023_men_baseline_bearded_delta_current_face_pp](figures/feature2023_men_baseline_bearded_delta_current_face_pp.png)

![feature2023_additive_fixed](figures/feature2023_additive_fixed.png)

![feature2023_applied_norm](figures/feature2023_applied_norm.png)

![feature2361_men_all_delta_current_face_pp](figures/feature2361_men_all_delta_current_face_pp.png)

![feature2361_men_baseline_bearded_delta_current_face_pp](figures/feature2361_men_baseline_bearded_delta_current_face_pp.png)

![feature2361_additive_fixed](figures/feature2361_additive_fixed.png)

![feature2361_applied_norm](figures/feature2361_applied_norm.png)

![feature4228_men_all_delta_current_face_pp](figures/feature4228_men_all_delta_current_face_pp.png)

![feature4228_men_baseline_bearded_delta_current_face_pp](figures/feature4228_men_baseline_bearded_delta_current_face_pp.png)

![feature4228_additive_fixed](figures/feature4228_additive_fixed.png)

![feature4228_applied_norm](figures/feature4228_applied_norm.png)

![feature4338_men_all_delta_current_face_pp](figures/feature4338_men_all_delta_current_face_pp.png)

![feature4338_men_baseline_bearded_delta_current_face_pp](figures/feature4338_men_baseline_bearded_delta_current_face_pp.png)

![feature4338_additive_fixed](figures/feature4338_additive_fixed.png)

![feature4338_applied_norm](figures/feature4338_applied_norm.png)

![feature4399_men_all_delta_current_face_pp](figures/feature4399_men_all_delta_current_face_pp.png)

![feature4399_men_baseline_bearded_delta_current_face_pp](figures/feature4399_men_baseline_bearded_delta_current_face_pp.png)

![feature4399_additive_fixed](figures/feature4399_additive_fixed.png)

![feature4399_applied_norm](figures/feature4399_applied_norm.png)

![feature4554_men_all_delta_current_face_pp](figures/feature4554_men_all_delta_current_face_pp.png)

![feature4554_men_baseline_bearded_delta_current_face_pp](figures/feature4554_men_baseline_bearded_delta_current_face_pp.png)

![feature4554_additive_fixed](figures/feature4554_additive_fixed.png)

![feature4554_applied_norm](figures/feature4554_applied_norm.png)

![feature4586_men_all_delta_current_face_pp](figures/feature4586_men_all_delta_current_face_pp.png)

![feature4586_men_baseline_bearded_delta_current_face_pp](figures/feature4586_men_baseline_bearded_delta_current_face_pp.png)

![feature4586_additive_fixed](figures/feature4586_additive_fixed.png)

![feature4586_applied_norm](figures/feature4586_applied_norm.png)

![feature5094_men_all_delta_current_face_pp](figures/feature5094_men_all_delta_current_face_pp.png)

![feature5094_men_baseline_bearded_delta_current_face_pp](figures/feature5094_men_baseline_bearded_delta_current_face_pp.png)

![feature5094_additive_fixed](figures/feature5094_additive_fixed.png)

![feature5094_applied_norm](figures/feature5094_applied_norm.png)

![Ten multiplicative features](figures/ten_feature_multiplicative.png)

![Ten additive features](figures/ten_feature_additive.png)

Women parser-comparison panels, fixed-baseline denominator panels, all PNG/PDF figures and seed30 paired galleries are in figures/. All arrays, original PNGs and masks remain on the research host. Existing exact-image predictions retain missing probability fields explicitly; fresh predictions preserve full class maps and lossless float32 beard-probability maps. No source images were regenerated, resized, or overwritten. Contact sheets alone use resized previews.

## Reproduce and resume

```bash
# Original smoke test: feature4228,12 conditions; expanded runs reuse its verified result.
# Shards: 0 through 7
sbatch config/code/score.sbatch /n/fs/goose/experiments/mid0_sae_selectivity/beard_all_features_20261117_021807_655293 --shard 0
PYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python /n/fs/goose/experiments/mid0_sae_selectivity/beard_all_features_20261117_021807_655293/config/code/analyze.py --run /n/fs/goose/experiments/mid0_sae_selectivity/beard_all_features_20261117_021807_655293
```

Completed assets are SHA-verified and skipped on resume. Each shard has an exclusive lock and append-only journal. Partial attempts are retained. Missing predictions block the final analysis rather than silently disappear. Input hashes, source CSV snapshots, checkpoint hashes, runtime versions and code are in config/. Full archive: `/n/fs/goose/experiments/mid0_sae_selectivity/beard_all_features_20261117_021807_655293`.


## GitHub release scope

Large per-image tables and input manifests are losslessly gzip-compressed (`.csv.gz`); read with pandas.read_csv or gzip.open. Private per-case distortion-review joins and reviewer labels are excluded. The full masks/probabilities/originals remain in `/n/fs/goose/experiments/mid0_sae_selectivity/beard_all_features_20261117_021807_655293`. The local full-output manifest lists those files; they are not all uploaded here.
