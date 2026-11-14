# Speaker notes — SAE racial-bias audit

20 slides; approximately 15–20 minutes.

The PNG previews visualize the authored layout; they are not a PowerPoint-engine render.

## Slide 1: Sparse autoencoders for a racial-bias audit

This presentation describes work that was actually performed on local artifacts. The central finding is methodological: the pipeline supports exact matched-seed replay and controlled feature scaling. It does not establish that racial bias or image quality improved. The portrait study uses the existing pretrained SAE so old feature IDs retain their reference. A separate small SAE training pilot is included only in the appendix. All race and gender labels refer to generation prompts, not inferred identities.

Local evidence: `FINDINGS.md; sae_bias_experiment.py`

## Slide 2: Three questions require different evidence

We distinguish association, intervention, and fairness evaluation. Association measures whether SAE activations vary with generation-prompt categories. The intervention tests what changes when a specific internal feature contribution is multiplied while holding the prompt and random seed fixed. A fairness claim additionally requires a predefined unwanted outcome, annotations, and a comparison of outcome rates across groups. The current run completes the first two stages for selected features; it has not completed the third.

Local evidence: `FINDINGS.md; sae_bias_experiment.py`

## Slide 3: Two existing datasets, two distinct roles

The baseline has 50 images per race/gender category, for 600 PNGs total. The separate cached feature file also contains 600 samples, each represented by 5,120 spatial-max-pooled pretrained SAE activations. It uses randomly sampled seeds from its own generation script. That file is used for feature discovery and held-out association reporting; baseline seeds 40, 41, and 42 are used for visual interventions. The cached codes are not the raw 1,280-channel residual inputs needed to train a new SAE. The cached file does not store per-row seed metadata.

Local evidence: `baseline/images/; race_classifier/race_classify_features.npz; race_classifier/generate_data.py`

## Slide 4: The baseline spans all 12 prompt categories

This slide provides the category coverage, not an assessment of the depicted people’s identities. Rows correspond to man and woman prompts. Columns follow the six requested race labels. The template fixes expression, shirt color, background, and a request for no facial coverings. The original phrase does not explicitly exclude a headscarf that leaves the face visible. This distinction matters when interpreting the Middle Eastern woman example. We replay this exact template rather than changing it during the intervention experiment.

Local evidence: `baseline/images/*/baseline40.png; baseline/baseline-image-generation.ipynb`

## Slide 5: What the SAE represents

The analysis taps unet.down_blocks.2.attentions.1. For every spatial position, the training representation is the attention block output minus its input, with 1,280 channels. The existing SAE expands this into a 5,120-dimensional feature dictionary using Top-K encoding with k=10 and ReLU. Each decoder column is a learned residual direction. The pretrained checkpoint also has an auxiliary-k setting of 256 in its configuration. We do not retrain this checkpoint in the portrait audit. This preserves feature IDs from the previous notebooks.

Local evidence: `sdxl-unbox/SAE/sae.py; pretrained down.2.1 final/config.json; sae_bias_experiment.py`

## Slide 6: Feature scaling preserves reconstruction error

Let r be the original residual and r-hat its SAE reconstruction. The residual reconstruction error is e = r minus r-hat. Scaling feature f from z-f to s times z-f changes the reconstruction by (s minus 1) times z-f times decoder direction d-f. Adding this delta directly to the original block output preserves e. This prevents generic SAE reconstruction artifacts from being mixed into every intervention. Strength one is therefore a mathematical no-op. Inactive features remain inactive under multiplicative editing; this procedure is not an additive feature injection.

Local evidence: `sdxl-unbox/scripts/sae_experiment.py: intervention; sae_bias_experiment.py`

## Slide 7: Candidate ranking controls for prompt gender

For each race/gender cell, an RNG with seed zero produces a 40/10 discovery/held-out split. For a given race and feature, we calculate the difference between that race’s discovery mean and the mean of other races within each gender. We divide by the feature’s standard deviation across all discovery samples, clipped at 1e-6 for numerical stability, and average the two gender contrasts equally. Candidate 3530 is the strongest additional gender-sign-consistent discovery association. Features 138 and 527 were included because of prior analyses. Held-out effects do not select the candidate. This is exploratory; no multiple-testing-adjusted significance is reported.

Local evidence: `sae_bias_experiment.py: analyze; feature_associations.csv; cached_data_split.npz`

## Slide 8: Prompt-associated features persist on held-out data

The chart shows the strongest discovery association for each requested race label and its corresponding held-out contrast. The numerical values are standardized effect contrasts, not classification accuracy or bias scores. All six displayed features have a positive association with their own prompt label in both splits. Only 527 and 3530 from this list were intervened on; feature 138 is an additional previously studied candidate. The other four top features remain association findings. The same prompt template appears in discovery and held-out data, so this does not demonstrate transfer to new prompt formulations.

Local evidence: `feature_associations.csv`

## Slide 9: A fully matched intervention experiment

The experiment includes 12 categories, three seeds, three features, and four strengths, yielding 432 edited images. The three features are 138, 527, and 3530. Replay uses the original fp16 VAE, Euler ancestral scheduler with trailing timesteps, one denoising step, guidance scale zero, and separately seeded latent and scheduler random streams. The original baseline and the new replay matched exactly for all 36 cases. Every strength-one image was also pixel-identical to its replay. This validates reproducibility and the no-op control, but does not measure fairness or quality. The current run generated 36 category/feature grids.

Local evidence: `settings.json; intervention_metrics.csv; sae_bias_experiment.py`

## Slide 10: Feature 138 does not cleanly control head coverings

The side-by-side images show two seeds from the Middle Eastern woman prompt category. Seed 40 contains a visible headscarf; removing feature 138’s contribution does not remove it. Seed 41 has visible uncovered hair, and the feature is active there as well. Scaling produces changes in face and hair detail, which agrees with the earlier notebook’s plain-hair confound. Heatmap intensity is normalized independently per image, so colors do not support cross-image magnitude comparisons. The original prompt prohibits facial coverings, not all head coverings; this example is not by itself an established prompt violation.

Local evidence: `middle_eastern_woman/feature138_comparison.png; individual seed PNGs; intervention_metrics.csv`

## Slide 11: Feature 3530 is associated, but not a scarf switch

Feature 3530 was selected as the strongest additional gender-sign-consistent discovery association after including 138 and 527. Its held-out standardized contrast for the Middle Eastern prompt label is 2.65. Despite that association, the intervention at strength zero leaves the scarf in seed 40. The visible changes are smaller and involve facial or local detail in these examples. This separates prompt association from control of a specific attribute. It does not imply the feature lacks effects under other prompts, strengths, blocks, or denoising steps; those were not tested here.

Local evidence: `middle_eastern_woman/feature3530_comparison.png; individual seed PNGs; intervention_metrics.csv`

## Slide 12: Feature 527 changes depiction, not an evaluated bias score

The two examples use the Black man prompt category. Feature 527 has a strong discovery and held-out association with the Black prompt label. Reducing or amplifying its contribution changes face texture and contrast. In seed 42, amplification also produces visible differences in composition and facial detail. These visual observations do not establish a semantic feature label or a quality preference. Suppressing a group-associated feature can alter a requested depiction without improving fairness. No identity inference or race classifier is applied to the edited images in this experiment.

Local evidence: `black_man/feature527_comparison.png; individual seed PNGs; intervention_metrics.csv`

## Slide 13: Intervention magnitude varies across prompt groups

The table aggregates the recorded mean absolute pixel differences between original and edited PNGs at strength zero. Each cell averages six images, two genders times three seeds. Values are on an 8-bit RGB scale from zero to 255. This quantifies whether an intervention has an effect and how large its image-level change is, not whether it improves quality or fairness. Feature 138 and 3530 are inactive in several tested categories, giving zero changes. Feature 527 also changes some Hispanic, Indian, and Middle Eastern samples. Results are descriptive with a small sample and no uncertainty intervals.

Local evidence: `intervention_metrics.csv; test_activation_metrics.csv`

## Slide 14: What the evidence supports—and what it does not

The experiments support exact baseline replay, correct multiplicative editing, associations that recur in a held-out cached split, and qualitative changes from selected feature interventions. They do not establish semantic race features, headscarf-specific control, mitigation of racial bias, improvement in image quality, or generalization beyond the current prompt template. An older classifier achieved perfect held-out classification on its own split, but this is prompt-group decodability, not a fairness score, and no classifier was applied to edited images in the present run. The distinction should be central to the research narrative.

Local evidence: `FINDINGS.md; sae_bias_experiment.py`

## Slide 15: Threats to validity

The intervention sample has only three seeds per category and uses one template, one block, and one denoising step. Candidate selection is partly informed by older analyses, so the held-out cached split is exploratory rather than a pristine confirmatory test. The spatial-max feature representation can emphasize a single location and does not directly locate a semantic concept. The intervention is restricted to features that already activate. Human reviews remain blank, and no inter-rater agreement, confidence intervals, or downstream quality measures have been computed. Overlay maps are normalized independently and should not be used for magnitude comparisons.

Local evidence: `FINDINGS.md; sae_bias_experiment.py`

## Slide 16: The next study should target a defined unwanted outcome

This slide proposes future work, not completed results. For a head-covering target, use a common explicit no-head-covering instruction for every category, along with a paired prompt variant. Reserve common seeds across all categories for final evaluation before candidate selection. Collect residual vectors from multiple blocks and compare matched activation changes, then evaluate feature interventions on those untouched seeds. Annotate unwanted coverings, hair damage, clothing compliance, image artifacts, and image preference with blinded or randomized presentation. Report within-group outcome changes and between-group disparities with uncertainty. Prompt compliance and requested appearance should be assessed alongside the bias target.

Local evidence: `FINDINGS.md; sae_bias_experiment.py`

## Slide 17: The experiment is reproducible and reviewable

The new bias experiment script has an analysis-only mode and a complete replay mode. Outputs include the dataset split, candidate associations, test activation metrics, intervention pixel changes, per-image raw maps, individual PNGs, 36 grids, and a markdown report. The annotation sheet has 432 rows, one per intervention, with blank fields for outcomes and preference; those blanks must not be mistaken for results. Existing notebooks and baseline images were preserved. The separate generic SAE training script remains available but its prompt split is not suitable as a balanced portrait audit without adaptation.

Local evidence: `sdxl-unbox/scripts/sae_bias_experiment.py; SAE_BIAS_EXPERIMENT.md; output REPORT.md`

## Slide 18: Appendix · the initial SAE training pilot

Before the portrait scope was clarified, a separate Top-K SAE was trained on ten varied prompts with four seeds per prompt, with two additional prompts held out. The actual dataset has 10,240 training vectors and 2,048 validation vectors; the selected block has 256 spatial positions per image, below the collection cap of 512. The new dictionary has 5,120 features and k=20 and was trained for 500 Adam steps at learning rate 1e-4. Held-out normalized MSE fell from 1.221 to 0.976, near a held-out-mean prediction baseline. 86.6 percent of features were unused on that validation sample, not globally dead. This pilot demonstrates trainability, not adequate feature quality, and its IDs do not transfer to the pretrained dictionary.

Local evidence: `baseline/sae_experiment_20261009/RESULTS.md; training_metrics.json; sae_experiment.py`

## Slide 19: Appendix · a normalization issue in the old notebook

The old race-prevalence notebook sums active spatial positions for each feature and divides by the number of generated images. Its comment calls the result a fraction of positions, but the value is the mean number of active positions per image. A true spatial active fraction divides by the total number of positions, or averages a per-image fraction when shapes are constant. The new cached-data association uses spatial maxima because that is what the existing feature file contains, and labels it accordingly. The new replay metrics calculate spatial means, maxima, and actual active fractions. No historical JSON, CSV, or notebook output was overwritten.

Local evidence: `baseline/sae_race_feature_analysis.ipynb; existing_feature_138_527.csv; test_activation_metrics.csv`

## Slide 20: Appendix · local evidence and exact configuration

The slides are grounded in local output files, rather than paper claims or external benchmarks. The pretrained checkpoint config lists 1,280 input dimensions, 5,120 features, k=10, auxk=256, and a dead-steps threshold of 2,441. The run uses cached SDXL Turbo and the fp16-fixed VAE, a single GPU, one 512-square denoising step, and the settings shown here. The working Python environment was isolated with -I to avoid a broken user-level torch import. Model startup initially waited on filesystem reads; after loading, generation and intervention checks completed. These environment details explain execution but are not research results.

Local evidence: `FINDINGS.md; settings.json; intervention_metrics.csv; pretrained final/config.json; build_sae_presentation.py`
