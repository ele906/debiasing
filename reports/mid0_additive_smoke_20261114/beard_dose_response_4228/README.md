# Existing-image beard-parser intervention scoring

{
  "n_conditions": 7440,
  "n_unique_assets": 4440,
  "n_new_generated_images": 0,
  "n_trained_models": 0,
  "features": [
    4228
  ],
  "n_private_review_joins": 10,
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

## Figures

![feature4228_men_all_delta_current_face_pp](figures/feature4228_men_all_delta_current_face_pp.png)

![feature4228_men_baseline_bearded_delta_current_face_pp](figures/feature4228_men_baseline_bearded_delta_current_face_pp.png)

![feature4228_additive_fixed](figures/feature4228_additive_fixed.png)

![feature4228_applied_norm](figures/feature4228_applied_norm.png)

Women parser-comparison panels, fixed-baseline denominator panels, all PNG/PDF figures and seed30 paired galleries are in figures/. All arrays, original PNGs and masks remain on the research host. Existing exact-image predictions retain missing probability fields explicitly; fresh predictions preserve full class maps and lossless float32 beard-probability maps. No source images were regenerated, resized, or overwritten. Contact sheets alone use resized previews.

## Reproduce and resume

```bash
# Original smoke test: feature4228,12 conditions; expanded runs reuse its verified result.
# Shards: 0 through 3
sbatch config/code/score.sbatch /n/fs/goose/experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166 --shard 0
PYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python /n/fs/goose/experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166/config/code/analyze.py --run /n/fs/goose/experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166
```

Completed assets are SHA-verified and skipped on resume. Each shard has an exclusive lock and append-only journal. Partial attempts are retained. Missing predictions block the final analysis rather than silently disappear. Input hashes, source CSV snapshots, checkpoint hashes, runtime versions and code are in config/. Full archive: `/n/fs/goose/experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166`.


## GitHub release scope

Large per-image tables and input manifests are losslessly gzip-compressed (`.csv.gz`); read with pandas.read_csv or gzip.open. Private per-case distortion-review joins and reviewer labels are excluded. The full masks/probabilities/originals remain in `/n/fs/goose/experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166`. The local full-output manifest lists those files; they are not all uploaded here.
