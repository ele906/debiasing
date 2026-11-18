# Independent SDXL Turbo SAE interventions at up.0.0

The two independently selected feature sets had similar MediaPipe detection rates through most shared perturbation doses. At the strongest shared doses, up.0.0 retained detection more often for feature decreases and less often for amplification. Its discovery activations were less concentrated inside face boxes, and its interventions generally changed the face less relative to the surrounding image. A separately screened up.0.0 feature, 3742, produced directionally consistent beard-area changes at moderate strengths.

## Hook, training, and discovery

The output hook is `unet.up_blocks.0.attentions.0`. At 512 × 512, both layers produced a [1, 1280, 16, 16] attention tensor. The SAE encoded the attention-module output minus its input, in channel-last layout. The matching pretrained up.0.0 Top-K SAE uses 5,120 directions, k=10, and auxk=256. The fine-tuned weight SHA-256 is `871cf2675d9bcb1b74c8e76779461f327efaf826fdcf7d4954cb5daa100080d0`. [Audit](AUDIT.md) · [Fine-tuned checkpoint hashes](training/checkpoint.sha256)

Training replayed 1,200 prompt/seed pairs with the mid.0 face-box methodology. The validation face MSE selected epoch 20. Across 179 held-out portraits with valid training-method face boxes, pretrained to fine-tuned MSE went from **7.401 to 5.917 inside faces (−20.1%)**, 5.802 to 5.511 outside, and 6.308 to 5.639 across the whole grid. The corresponding mid.0 face MSE went from 4.866 to 4.457. Mean discovery residual RMS was 5.172 at up.0.0 and 3.625 at mid.0. Residual scales differ despite identical tensor shape. [Training curves](training/training_curves.png) · [Training config](training/settings.json) · [Reconstruction results](training/heldout_reconstruction.json)

The table uses seeds 100–199 and the historical candidate filter, the union of each group's ten highest mean features. These ten also occupied the first ten rows of the full dictionary ranking. Frequency denotes positive SAE activation over all discovery tokens.

| up.0.0 feature | Peak prompt group | Peak minus other means | Frequency |
| --- | --- | ---: | ---: |
| 1088 | Middle Eastern woman | 18.51 | 5.73% |
| 4099 | Middle Eastern woman | 17.87 | 5.95% |
| 55 | East Asian woman | 13.08 | 2.63% |
| 2261 | Middle Eastern man | 12.90 | 4.88% |
| 4303 | Black woman | 10.27 | 3.06% |
| 4063 | White woman | 8.52 | 4.28% |
| 3742 | Middle Eastern man | 7.79 | 3.17% |
| 3103 | Middle Eastern woman | 7.35 | 5.96% |
| 93 | Indian woman | 7.29 | 2.52% |
| 2655 | Middle Eastern woman | 7.24 | 9.53% |

[Full ranking with activation magnitudes](discovery/full_ranking.csv) · [JET overlays](discovery/overlays) · [Per-image matched discovery maps summary](discovery/spatial_by_image.csv)

[Per-feature prevalence histograms and multi-seed JET galleries](FEATURE_GALLERIES.md)

## Controls and stability

The 1,260-condition pilot covered features 1088, 4099, and 55 at α = −30, −7, 0, 1, 3, 10, and 30. Pixel replay passed at α=1 and −7. The pilot passed before SLURM submitted the full run. The full run recorded 17,400 unique conditions, including 600 α=1 controls. Every clean control matched the archived mid.0 sweep control at the pixel level. The older source dataset PNGs differed from these controls. All 9,106 zero-activation conditions were pixel-exact no-ops. [Pilot gate](pilot/validation.json) · [SLURM jobs](jobs.txt) · [Source manifest](source_manifest.sha256) · [Full shards](full)

Among nonzero applied edits, MediaPipe detected a face in 96.1% of up.0.0 images and 98.4% of mid.0 images. Median applied L2 was 1,231 and 860 respectively. Median residual-relative L2 was 0.415 and 0.417. These pooled rates summarize different dose and feature mixtures. All 600 controls per layer were detected.

For dose matching, shared log-spaced bins used measured post-cast L2 or L2 divided by the original residual norm. Tensor RMS is raw L2 divided by √327,680 in both layers. Prompt/seed pairs received equal weight inside each bin, with uncertainty summarized across five seeds. Both layers had 100% detection in the shared relative-dose bins through approximately 2.0 in both directions. In the strongest normalized bins, detection at up.0.0 versus mid.0 was **48.1% versus 30.1%** for decreases and **35.5% versus 74.1%** for amplification. The corresponding differences were +18.0 ± 5.0 and −38.6 ± 12.6 percentage points (mean ± seed SEM, 28 pairs each). With raw L2 matching the strongest-bin differences were +43.1 ± 6.8 and −7.2 ± 8.9 points. The amplification contrast is sensitive to dose normalization, and the highest bins are broad. [Dose-matched curves](full/evaluation/dose_matched_curves.png) · [Bin estimates](full/evaluation/dose_matched_curves.csv) · [Feature curves](full/evaluation/up.0.0_detection_by_feature.png) · [Prompt-group curves](full/evaluation/up.0.0_detection_by_group.png) · [Sampled failures](full/evaluation/sampled_detection_failures.csv)

MediaPipe measures detector success. Of 22 cases in the matched mid.0 sweep with existing majority human ratings, seven were rated distorted despite successful detection. The ratings came from a selected subset. Up.0.0 has no human labels. RGB MAE measures image change, including intended edits. [Human-review aggregate](full/evaluation/human_review_counts.csv) · [Per-image evaluation](full/evaluation/per_image.csv)

## Localization and visible changes

Across independently selected top-ten features, mean positive-token frequency was 4.77% at up.0.0 versus 4.10% at mid.0. When a feature was active and a face box was available, 19.2% of its activation mass fell in the up.0.0 face box versus 47.3% at mid.0. The boxes covered roughly 31% of the grids. These measurements describe the selected tokens and approximate detector boxes. [Discovery spatial summary](full/evaluation/discovery_spatial_by_feature.csv)

The pixel comparison used fixed convex hulls from the baseline face landmarks. At relative L2 around 0.43, up.0.0 changed pixels inside the facial region by roughly 3.3–3.7 fewer RGB intensity units, with outside-region differences of roughly +0.2–0.5 units. This supports **less face-concentrated** image change for the up.0.0 feature set. Spatial compactness outside the face remains unresolved. The hull excludes most hair, scarves, and clothing. Gauge the result alongside the activation and edit galleries. [Localization contrast](full/evaluation/dose_matched_curves.csv) · [Target-context edits](full/evaluation/gallery/target_contexts.jpg) · [Per-feature edit galleries](full/evaluation/gallery)

In selected examples, 1088 and 2261 changed head coverings, 4099 also changed hair and background texture, and 3742 changed beard extent and sometimes headwear. Strong edits could produce broad structural changes. These are visual observations and candidates for controlled semantic tests. Feature IDs and meanings are independent across layers.

The sweep saved the fraction of newly positive dictionary entries after re-encoding and the decoder-space distance between intended and re-encoded codes. For active up.0.0 edits the median distance rose from 75 at α=0 to 189 at α=7 and 3,329 at α=100. Direct multiplication operates on existing support and adds no positive entries to the intended edited code. The archived mid.0 support-expansion experiment used a different intervention design, so these diagnostics are reported for up.0.0 without a matched cross-layer effect estimate. [Sharded intervention records](full)

## Feature 3742 beard follow-up

The existing DINOv3 FaceParsing checkpoint scored 816 unique images across the 1,740 conditions for this single feature. Relative to the fixed baseline face denominator, mean beard-area changes at α=−7 and +7 were **−1.92 and +4.03 percentage points** across all 60 images and **−6.02 and +12.50 points** among the 20 baseline-bearded images. At α=+7 in the five Middle Eastern man prompt samples, the mean change was +22.69 points. At extreme strength the response became nonmonotonic and the galleries showed visible corruption.

Within a shared applied L2 bin centered near 1,061 for amplification, the fixed-denominator mean change was +14.13 points for up.0.0 3742 and +6.29 for mid.0 4228, over 18 shared prompt/seed pairs. The parser is a proxy and the two features were chosen independently. The additive follow-up was deferred pending validation of beard onset and off-target effects on baseline-clean portraits. [Paired scores](beard/paired.csv) · [Denominator and subgroup curves](beard/beard_response.png) · [Matched L2 comparison](beard/matched_mid0_4228.png) · [Baseline, edit, and mask](beard/baseline_edit_mask.png)

## Limitations and next experiments

- Five evaluation seeds and layer-specific feature selection limit causal attribution to layer placement. Repeat with held-out feature discovery and prespecified raw and residual-relative L2 targets, including finer dose spacing near first sampled failures.
- Discovery overlaps SAE fitting. The prompt-group means measure associations with generated prompt categories. Use matched attribute prompts, blinded structural ratings, and masks for headwear, hair and lower-face regions to test semantic specificity.
- Score facial hair and off-target changes in controlled baseline-clean and baseline-bearded contexts before testing additive beard onset. Pair additive and multiplicative edits by applied dose and vary the injection footprint independently.
