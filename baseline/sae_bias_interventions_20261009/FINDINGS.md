# Findings from the baseline SAE experiment

Used the six original generation-prompt labels crossed with man/woman, not
race inferred from image pixels. The existing baseline contains 600 PNGs.
The separate cached dataset contains 600 spatial-max-pooled SAE feature vectors.

## What ran

Used the original pretrained down.2.1 SAE, preserving feature IDs from the
previous notebooks. The unrelated new SAE pilot was not used. Ranked features
using 480 cached samples with gender-controlled standardized contrasts and
reported effects separately on 120 held-out cached samples.

Replayed baseline seeds 40, 41, and 42 in all 12 categories. Tested features
138, 527, and 3530 separately at strengths 0, 0.5, 1, and 2, preserving the SAE
reconstruction error. Saved 432 edited PNGs and 36 comparison grids.
All 36 original replays were pixel-identical to the saved baseline PNGs.
All 108 strength-1 images were pixel-identical to their unmodified replays.

## Associations in the cached data

Top discovery feature for each prompt race, averaging the within-gender
one-versus-other-races contrast, standardized by discovery feature variation:

| Prompt label | Feature | Discovery contrast | Held-out contrast |
|---|---:|---:|---:|
| Black | 527 | 2.44 | 2.28 |
| White | 4384 | 1.94 | 1.87 |
| East Asian | 1431 | 1.63 | 1.41 |
| Middle Eastern | 3530 | 2.44 | 2.65 |
| Indian | 2527 | 2.43 | 2.47 |
| Hispanic | 1678 | 1.72 | 1.64 |

Feature 138 also has a strong Middle Eastern prompt association (2.40 discovery,
2.54 held-out). These are feature-level associations with generation prompts,
not evidence of unfairness, semantic labels, or human racial identity.
Only features 138, 527, and 3530 were intervened on in this run.

## What the image edits show

In the Black-man grid, feature 527 affects face texture/contrast, with visible
changes in pose/crop at some amplified settings. Removing group-associated
features does not automatically remove an unwanted attribute.

In the Middle-Eastern-woman grid, seed 40 depicts a headscarf with the face
visible. The original template says "without any facial coverings"; it does
not explicitly forbid headscarves. Reducing feature 138 to zero does not remove
that scarf. It affects facial/hair detail in the three examples, consistent with
the previous plain-hair confound. Feature 3530 suppression also leaves that
scarf in place. Neither intervention demonstrated a clean headscarf-removal
control at the tested strengths and block.

The experiment establishes usable feature controls and exact reproducibility.
It does not establish reduced racial bias or improved image quality.

## How to use this for a bias study

Choose a concrete outcome before selecting an intervention: unwanted coverings
under an explicit no-head-covering prompt, clothing mismatch, image artifacts,
or another unwanted depiction. Use the same outcome and prompts across groups.
Annotate baseline and edited images, keeping requested appearance separate from
undesired artifacts. The supplied `human_review.csv` has blank fields for this.

The next useful head-covering experiment would use paired prompts with and
without an explicit no-head-covering instruction, multiple seeds, and multiple
blocks, then validate candidate feature effects on separate seeds. This is more
specific than suppressing features that merely distinguish prompt race groups.

For a newly trained SAE, collect balanced raw block residuals and reserve common
seeds across all 12 categories for validation. Existing pooled SAE codes are not
raw training inputs, and old feature IDs will not transfer to a new dictionary.

See [all grids](REPORT.md), [feature associations](feature_associations.csv),
[image-change metrics](intervention_metrics.csv), and [human review sheet](human_review.csv).
