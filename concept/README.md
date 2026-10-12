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
