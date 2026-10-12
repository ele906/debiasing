# Concept discovery and SAE intervention pilot

Research question: can visible concepts associated with harmful racial depictions be linked to SAE features whose attenuation reduces those depictions while preserving ordinary representation and image quality?

This starter adapts the concept auditing idea from [Concept2Concept](https://arxiv.org/abs/2410.04634). It uses manual concept discovery first; it does not yet implement the paper's automatic detector or claim to replicate its results. No Florence-2 checkpoint was present in the inspected model cache. Existing portraits are not a labeled racist-image dataset.

## Existing setup

The source is `../backprop`: 1,200 SDXL Turbo portraits, prompt/seed splits, and pretrained SAE pooled activations. The SAE uses Top-K sparsity (k=10), 5,120 features, and the residual at `unet.down_blocks.2.attentions.1`. It is not the L1 SAE from the pasted feature-suppression post. Generation settings and checkpoint hash are copied into the pilot provenance.

## Start reviewing

Open `pilot/gallery.html` and edit `pilot/review.csv`. The initial pilot contains three discovery and three validation images per race/gender prompt cell (72 images). Test images are reserved. Increase `--per-cell` with a new output directory for stronger controls.

- `concepts`: semicolon-separated visible descriptions, consistently named; `NONE` means reviewed with no concepts, while blank means unreviewed. Record ordinary concepts as well as candidate harmful ones.
- `harmful_depiction`: blank until reviewed; then `0`, `1`, or `uncertain` with `harm_reason`. A visible object, clothing item, or racial identity is not itself a harm label.
- `reviewer` and `notes`: document judgments and ambiguity. Define the labeling rubric with your professor before treating these as ground truth.

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python concept/pilot.py rank
```

Ranking compares concept-present versus concept-absent images within each race/gender prompt group, averages qualifying group contrasts, selects features on train only, and reports validation differences. Groups need at least two examples of each class. No eligible controls means no ranking, not a negative finding. These are descriptive associations; inspect SAE maps and additional controls before calling them concept features. Human review labels and detector predictions should be kept distinct when automatic extraction is added.

## Later intervention

Reuse the existing residual-preserving intervention in `../backprop/fixed_suppression.py`. Once IDs are validated, run with those IDs and an output under this folder:

```bash
# Replace FEATURE_ID with a validated integer; requires a GPU.
/n/fs/goose/el8403/conda-envs/reno/bin/python backprop/fixed_suppression.py \
  --features FEATURE_ID --strengths 0 0.5 1 --output concept/intervention_001
```

Strength 1 is unchanged, 0.5 attenuates, and 0 removes the selected SAE contribution. The pretrained SAE and diffusion weights remain frozen. Include matched random-feature controls in a subsequent run. Compare identical prompts/seeds and review harmful depiction, identity preservation, prompt fidelity, and artifacts. Lower feature activation or pixel change alone is not reduced racism. The intervention script writes its own review sheet and verifies strength-1 identity.

## Status

Pilot preparation ran. Automatic extraction is implemented below; feature selection and intervention have not run. Next: review extracted concepts, agree on a concept/harm rubric, label discovery and validation controls, then rank and visually validate candidates.

## Automatic extraction

`extract.py` now runs the pinned `microsoft/Florence-2-base-ft` detector locally. It generates a detailed caption, grounds phrases from that caption, and performs object detection. This is a Concept2Concept-style adaptation, not a full replication of the paper. The detector was downloaded into `concept/hf_cache`; its exact revision is in `model_revision.txt`.

```bash
sbatch concept/extract.slurm
```

Summary outputs are at the top level of `concept/`; per-image JSON and grounding images are under `extracted/`:

- `REPORT.md`: frequent detector phrases and example captions.
- `concept_predictions.csv`: model predictions alongside blank human review fields.
- `concept_frequencies.csv`: image-level phrase counts (duplicate labels within an image counted once).
- Per-image JSON: captions, boxes, raw generation text, and image hashes.
- `*_grounded.jpg`: image with phrase bounding boxes.

Predictions may omit or hallucinate concepts. Correct/standardize them into the `concepts` column; leave `harmful_depiction` separate. Synonyms remain separate in the raw frequency report. To rank reviewed labels:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python concept/pilot.py rank \
  --review concept/concept_predictions.csv --output concept/reviewed_rankings
```

The script resumes completed images only when settings and image hashes match. Human corrections belong in a separate copy of the prediction CSV, since re-running extraction regenerates its CSV summaries. The input review sheet is never edited.

Florence-2 requires `timm==0.9.16`, installed without dependencies into `concept/deps`. The Slurm script places this folder on PYTHONPATH without changing the existing environment.

### First completed run

Job `4086269` completed extraction for all 72 images. Output integrity checks passed: unique sample IDs, one JSON and grounded image per sample, matched label/box counts, no test samples, and no automatically assigned human concept or harm labels. Examples include headscarf/beard, red turban, afro/earrings, and kimono. These are detector outputs requiring review; cultural clothing and hairstyles are not inherently harmful. Grounding can omit a concept mentioned in a caption, while object detection may still detect it. No SAE features have been selected or suppressed.

Model cache, dynamic modules, and isolated dependency files are excluded by the local `.gitignore`.

## SAE comparison with automatic concepts

Run `python concept/compare_sae.py` in the existing reno environment. It joins the 72 detector samples to the saved SAE spatial means by sample ID, verifies the SAE checkpoint hash, and normalizes detector phrases with explicit regex patterns. These provisional labels remain separate from human review labels. Discovery data select features; validation data report agreement. Within-prompt-group contrasts are preferred; global-only rankings are flagged as confounded. The tiny pilot permits groups with a single positive/negative example and provides exploratory evidence only.

Results: [SAE report](sae_comparison/REPORT.md), [summary](sae_comparison/summary.csv), and [top candidates](sae_comparison/candidates.csv). The headscarf candidate has validation AUC about 0.72. Several candidates are near chance. Kimono and flower choose the same feature (1899), with no within-group controls: perfect separation can reflect prompt-group associations, not separate concept features.

`sbatch concept/sae_maps.slurm` replays 12 discovery examples for headscarf, necklace, and kimono and saves original/map comparisons under `sae_comparison/maps`. Pixel replay must match the original exactly. No model or SAE is trained, and no features are suppressed. Visual inspection and stronger controls remain necessary before intervention.

## Category counts and bounding boxes

`category_counts.py` creates image-level normalized concept counts/rates, raw phrase counts, race-pooled counts, and boxed image sheets. The completed pilot is in [category_analysis/REPORT.md](category_analysis/REPORT.md): 12 categories, six images each, 739 unique boxes. The dataset has 12 prompt categories, not 16. All detected boxes have source labels and numbered legends; phrase grounding and object detection are both included. Category sheets show six examples per category; full-size individual panels stay local under boxes/.

## Full 1,200-image extraction

`full_dataset.py prepare` splits the existing manifest into four 300-image shards and reuses exact-hash-matching pilot predictions. `extract_full.slurm` runs the detector on all images, including test images for a descriptive audit. After all shards complete, `full_dataset.py merge` verifies image hashes and box/label counts, archives the 72-image summaries under `pilot/extraction_summary/`, and replaces the top-level extraction report and prediction/frequency CSVs with full-dataset results.

For the full SAE comparison, use `compare_sae.py --exclude-test --output concept/sae_comparison_full`: 720 training and 240 validation samples; the 240 test samples do not choose features. Full-dataset category counts can include test samples for descriptive auditing, so the dataset has now been inspected broadly; further confirmatory conclusions should use new independent controls.

The pilot SAE spatial replay completed with 12 exact image reproductions. Headscarf 3364 was inactive on the two selected detector-positive examples, and inspected necklace 43 activation was near the head rather than the necklace. Kimono 1899 overlapped clothing but remains prompt-group-confounded. The current candidates are exploratory rather than established intervention targets.

### Completed full results

All 1,200 extraction outputs were verified. Top-level `REPORT.md`, `concept_predictions.csv`, and `concept_frequencies.csv` now cover all 1,200 images. The original 72-image summaries are preserved in `pilot/extraction_summary/`.

- [Complete counts by category](category_analysis_full/counts_by_category.md)
- [Complete rates by category](category_analysis_full/rates_by_category.md)
- [Complete raw phrase counts](category_analysis_full/raw_phrase_counts_by_category.md)
- [Complete race-pooled counts](category_analysis_full/counts_by_race.md)
- [Full bounding-box report and category sheets](category_analysis_full/REPORT.md)
- [Full SAE comparison](sae_comparison_full/REPORT.md)
- [Complete SAE summary table](sae_comparison_full/summary.md)
- [All SAE candidates](sae_comparison_full/candidates.md)

Each of the 12 source categories has 100 images. Turban detector counts are 93/100 for Indian-man prompts, 27/100 for Middle-Eastern-man prompts, and 1/100 for Indian-woman prompts. Raw phrase tables preserve generic scarf labels separately; headscarf patterns do not treat every scarf as a head covering. Malformed reversed-coordinate boxes are kept in raw predictions but skipped in drawings.

The full SAE ranking uses train-only within-group ROC AUC, weighted by the smaller class size, requiring at least three detector positives and three negatives per qualifying group and at least five detector-positive training images on which the feature activates. This avoids promoting features that fire on just one image. Global AUC is a tie-breaker or flagged fallback when training group controls are unavailable. The report shows both global and within-group validation agreement. The 240 test rows are excluded before all SAE comparison calculations.

Necklace feature 3985 has global validation AUC about 0.805 and within-group AUC about 0.742, but only one qualifying validation group. Afro feature 1500 has high global agreement (about 0.966), with weaker within-group agreement (about 0.667). Beard feature 5085 has high global agreement but below-chance within-group agreement, demonstrating confounding. The full candidates differ from the pilot IDs and do not inherit the pilot's spatial validation. Detector review, fresh controls, and spatial/causal checks remain before suppression.

`markdown_tables.py` exports all rows and columns of count/rate and SAE result CSVs; wide concept tables are split into smaller column blocks without dropping data. The full detector records, dependency caches, and 1,200 individual full-size boxed panels stay local; the complete summary tables, box coordinate CSV, and 12 representative category sheets are staged for Git.

## Current intersection checks

[Quick update and presentation wording](QUICK_UPDATE.md). The current spatial checks supersede interpreting image-level agreement alone as a concept intersection.

`intersections.py` replays all 240 validation images and measures SAE activation inside matching C2C-style detector boxes, normalized by box area. It provides circular-translation reference scores and exact replay checks. The report distinguishes pooled localization from localization in the same qualifying prompt groups used for within-group validation. Matching controls require at least three detector positives and three negatives. Recorded visual review issues are kept separate from raw detector labels.

The top-one checks are in [intersections/REPORT.md](intersections/REPORT.md). All five train-selected candidates for necklace, afro, and beard are in [intersections_alternatives/REPORT.md](intersections_alternatives/REPORT.md). Numerical screening is not semantic validation: confirmed detector misses are flagged, and all alternative candidates are reported.

Necklace-region feature 3181 is the current provisional overlap. The earlier 3985 candidate failed the controlled spatial test. Afro and beard controls have observed detector-label errors; their specificity is unresolved. No weights were changed or features suppressed.

`fresh_controls.py` generates 48 independently seeded paired controls with fixed features 3181, 1500, and 2893. The manifest preserves requested conditions separately from blank visual-review labels. Requested present/absent prompts do not establish actual concept presence. These controls are additional data, not part of the original 1,200-image count tables.
