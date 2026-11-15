# Does Top-K support expansion independently predict intervention instability?

**Short answer:** this dataset supports design-dependent associations, especially with selected-coordinate re-encoding discrepancy. It does not identify support expansion as an independent driver of instability, widespread image change, or visual unnaturalness.

## Bounded scope

Only the existing 200 edited feature-4228 conditions are used: 40 natural, 20 uniform, 120 shuffled and 20 localized. One face-fine-tuned SAE (SHA-256 cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6), the same SDXL Turbo generator, and mid.0. No diffusion, model inference/training, image generation, or annotation requests occurred. The analysis reads saved tables and hash-verified latent archives on CPU.

The independent resampling units are five paired recipient seeds. Prompt groups, both strengths, the two fixed donor maps and all three shuffled replicates remain together when a seed is resampled. Each recipient × target × method cell receives total weight one; 120 shuffled images do not receive six times the weight of 20 uniform images.

## Exact metric definitions

| Metric | Definition | What it does not establish |
|---|---|---|
| Support expansion | Fraction of 256 spatial positions with >10 nonzero full-latent coordinates after intervention | Visual unnaturalness or semantic degradation |
| Recoding discrepancy | ||encoded_edited_feature4228 − edited_feature4228||₂ over the 16×16 map | Full-latent consistency or decoded reconstruction MSE |
| Pixel MAE | Mean absolute edited-minus-baseline RGB difference, 0–255 scale | Image corruption or quality loss |
| Lower-face concentration | Sum of per-pixel RGB MAE weighted by frozen soft lower-face ROI, divided by total per-pixel MAE | Beard coverage or image-wide geographic spread |
| Outside concentration | 1 − lower-face concentration | Whether changes are distributed throughout the entire image |

## 1. Can support be separated from layout?

Method alone explains **98.7738%** of balanced support variance. The primary paired-design model explains **99.2460%**, leaving only about **0.754%** as residual variation. Uniform support ranges from 0.9102 to 1.0000, with no overlap with natural (0–0.08984) or shuffled (0.02734–0.08984) support. Localized ranges from 0.0625 to 0.32031.

All 100 recipient/pattern pairs have exactly the same support at both perturbation targets. All original full-latent vectors have exactly k=10 nonzeros. Since only one coordinate receives nonnegative additions, the following identity holds exactly for every image:

**Expansion = nonzero-pattern fraction − overlap of the pattern with originally active feature-4228 positions.**

Once those two geometry components are included, support has no residual variation and its separate coefficient is not identifiable. This identity is a design audit, not proof that support has no causal effect; conditioning on a deterministic construction removes the constructed variable by definition. The experimental problem is lack of separately assigned support variation and common support across layouts.

## 2. Association estimates and sensitivity

Slopes are changes per **+1 percentage point** of support expansion; intervals resample whole seed clusters. They are descriptive conditional slopes, not causal effects or held-out predictive accuracy.

| Outcome | Target-only slope [95% CI] | Full design + actual-L2 rounding residual [95% CI] | Matched-target design without ppm jitter [95% CI] |
|---|---|---|---|
| post_reencoded_discrepancy | 0.229419 [0.176687, 0.287177] | 0.274412 [-0.00953748, 1.10177] | 0.264738 [0.0229355, 0.931298] |
| pixel_mae | 0.000957221 [4.85891e-05, 0.00179551] | -0.0153772 [-0.137838, 1.04695e-05] | -0.0157608 [-0.131823, -0.00151334] |
| lowerface_change_fraction | -0.000250896 [-0.000448506, -5.51354e-05] | -0.000207868 [-0.0020708, 0.0049464] | -0.000339186 [-0.00426088, 0.00482805] |

The positive pooled recoding association is partly retained under matched-target controls. Applied-L2 deviations are at most about 6.3 parts per million relative to target. Adding these tiny rounding deviations as a covariate changes interval precision, so a blanket statement that all adjusted associations disappear would be incorrect. A sensitivity allowing method/donor/shuffle effects to vary by magnitude and recipient × magnitude blocks produces the same matched-target coefficients here because support is invariant across strength in this balanced paired design.

Pixel MAE changes sign after adjustment: the slight positive pooled association becomes negative or uncertain. It does not support a general “more support expansion means more image disruption” conclusion. Lower-face/outside concentration has no clear adjusted relationship in either nuisance specification.

Within natural-pattern interventions, the fully adjusted selected-recoding slope is **+4.4106 per percentage point**, bootstrap CI **[3.4807,12.4998]**. This is a remaining conditional association, not evidence of an independently manipulated support effect. Shuffled edits have a much less certain adjusted recoding slope (**+3.6001**, CI **[−0.2267,7.1437]**). Uniform and localized support are constant within a recipient, so recipient fixed effects absorb them completely.

When restricted to originally inactive recipients, natural/shuffled support is determined by donor pattern size; donor controls remove its separate variation. The apparent within-natural relationship relies on overlap with naturally active contexts, not an independent support manipulation.

Some within-method bootstrap samples are non-identifiable: **32/3125 (1.024%)** for natural/shuffled primary adjusted fits. Their mass is explicitly reported; conditional percentile intervals omit only those non-identifiable fits. Leave-one-seed-out estimates and all alternative specifications are exported. With five clusters, none of these intervals is a robust population guarantee.

## 3. Mechanical coupling of the recoding metric

The saved selected-feature discrepancy obeys an exact decomposition:

**D² = Σ(post_i=0) edited_i² + Σ(post_i>0)(post_i−edited_i)².**

The first term is edited-coordinate mass that the re-encoder does not retain. Its mean share of squared error is:

| Method | Mean error-energy share where re-encoded feature is zero |
|---|---:|
| natural | 58.07% |
| uniform | 88.27% |
| shuffled | 91.65% |
| localized | 84.57% |

This directly couples the metric to which coordinates survive encoding and to their amplitudes. The table is an exact decomposition, not a claim that support expansion caused these percentages. High selected-coordinate error need not imply bad image quality.

For any edited full latent h′, a k-sparse vector has a latent-coordinate squared approximation error at least equal to the squared mass outside h′’s largest k coordinates. That lower bound is audited from saved tensors. It does **not** lower-bound the selected-coordinate metric: the encoder can retain feature 4228 and drop another coordinate. Nor does it lower-bound decoded reconstruction error in a nonorthogonal, overcomplete dictionary. Actual full post-encoding latents were not saved, so no full reconstruction-consistency measurement is invented.

## 4. Same support and norm, different responses

For the four initially inactive East-Asian-man seeds, natural and shuffled maps have identical support for each donor. At L2=65.1363, natural minus shuffled yields:

| Outcome | Paired difference [95% seed-bootstrap CI] | Seed clusters |
|---|---|---:|
| post_reencoded_discrepancy | -9.28149 [-11.2554, -7.30757] | 4 |
| pixel_mae | 0.114049 [0.0387568, 0.189341] | 4 |
| lowerface_change_fraction | 0.0116538 [-0.0106039, 0.0339116] | 4 |

These contrasts show that support count alone is insufficient to summarize responses. In this subset, natural maps recode differently from shuffled maps at the same support and matched magnitude; spatial concentration remains uncertain. The subset is defined by support equality, not by observed outcomes, but has only four seeds (and the analogous Black-man subset only two).

## Explicit answers

1. **Does support expansion predict re-encoding inconsistency?** It is positively associated with the selected-coordinate discrepancy in pooled data and within natural maps; some adjusted specifications retain that association. It is not an independently identified instability predictor: the outcome partly reflects Top-K coordinate competition, support is design-linked, and equal-support layouts still differ in recoding.
2. **Does it predict spatially widespread image changes?** No robust independent relationship is established. Adjusted lower-face/outside concentration intervals span zero. Pixel MAE has a weak pooled relationship whose direction reverses under adjustment. Neither metric proves visual instability or image-wide geographic spread.
3. **Do relationships remain after controlling for intervention method?** Some recoding associations do; their precision depends on nuisance controls. Uniform/localized within-recipient support effects are not estimable, shuffled effects are uncertain, and no consistent image-spread association remains. Controlling the exact geometry construction makes a distinct support coefficient non-identifiable.
4. **Meaningful finding or mainly design consequences?** The meaningful finding is a limitation of intervention design and measurement: layout nearly determines support, matching L2 does not remove that confounding, and a Top-K-derived recoding metric partly measures its own coordinate-selection mechanics. There is no evidence here for a universal causal support-to-instability law. All face detections succeeded, so the dataset also lacks a varying face-failure outcome.
5. **Smallest additional experiment?** Start with a tightly scoped, within-layout **support-restoration policy test**, not another broad sweep. On the four already-inactive East-Asian recipients, reuse one donor pattern and one target: compare the existing unrestricted edit with a feature-retaining k-support-restored version (remove one specified non-target active coordinate per newly activated position), plus a deletion-only sham. Preflight attainable norms using saved block tensors; record changes in coordinate amplitudes and match actual total perturbation magnitudes. Where feasible, this needs only two new arms × four contexts (eight new images), reusing the unrestricted arm. An activation-only counterfactual can be inspected first without diffusion, but cannot establish new image effects. This tests a specified restoration policy, not support count in isolation, because deletion changes other coordinates; that caveat must remain explicit. No additional experiment is executed here.

## Reproduction and preservation

Run `analysis.py --run THIS_DIRECTORY` in the recorded NumPy/Matplotlib environment to repeat the main analysis from copied CSVs and existing hash-verified archives. Then run `sensitivity.py THIS_DIRECTORY`, `tests.py`, and `finalize.py`. No neural model package is imported. Inputs and all 200 raw activation archive hashes are preserved, as are code snapshots, configuration and package/git state. Original experiments are untouched.

Figures 01–08 provide pooled and within-method scatters, support overlap/R², conditional residuals, mechanical coupling, specification sensitivity and same-support contrasts. Missingness, bootstrap non-identifiability, paired per-seed differences and repeated-donor weighting are explicit in the CSVs. All analyses are exploratory, with no multiplicity-adjusted confirmatory claim.
