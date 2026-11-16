# SAM ViT-B pilot across all race/gender prompt groups

**Existing images only; no new diffusion generation and no training.** This is a segmentation feasibility pilot, not a validated beard-area experiment.

Included: 36 saved feature-4228 images, all12 prompt groups, seed30, and baseline/+10/+50 conditions. Completed SAM prompt predictions: 72/72; all216 candidate masks and scores are retained.

## Pilot decision: not ready for automatic beard-area scoring

All72 prompt evaluations returned nonempty masks. Inspection of the complete baseline overviews shows that several proposals follow neck/chin skin, lips, jewelry or head-covering fabric rather than visible beard hair. This is a qualitative failure-mode diagnosis, not a measured false-positive rate: no pixel ground truth or formal beard-presence labels were supplied.

SAM remains useful as a prompt-assisted region segmenter. For example, the box-plus-points Indian-man proposal captures much of the visible white beard (quality0.934), but a white-woman chin/neck-skin proposal receives quality0.955. The model-quality score therefore cannot be used here as a beard-presence threshold. The box-only Indian-man proposal also selects a non-beard region, illustrating prompt dependence.

The observed failures concern this fixed prompt policy, not an estimated race/gender performance disparity. There is one seed per group, and visible facial-hair presence is not inferred from prompt gender. All twelve groups and all three source conditions are retained.

**Do not run all4,200 images through this policy and label the result beard coverage.** A beard-presence gate or explicitly hair-grounded prompts must first be checked on diverse examples. Clipping masks to the lower-face ROI removes some geometric leakage but does not distinguish skin from hair. Candidate agreement and quality scores cannot substitute for that semantic check.

A reasonable next step is a small, independently checked facial-hair presence/attribute scorer, with SAM retained for outlining confirmed visible hair. Alternatively, use SAM as an assistive mask tool with a few actual hair/background points on a reviewed subset. Neither additional training nor a new diffusion run was performed in this pilot.

### All-candidate baseline previews

[black_man: all three candidates per prompt](figures/candidates_black_man.png)

[black_woman: all three candidates per prompt](figures/candidates_black_woman.png)

[east_asian_man: all three candidates per prompt](figures/candidates_east_asian_man.png)

[east_asian_woman: all three candidates per prompt](figures/candidates_east_asian_woman.png)

[hispanic_man: all three candidates per prompt](figures/candidates_hispanic_man.png)

[hispanic_woman: all three candidates per prompt](figures/candidates_hispanic_woman.png)

[indian_man: all three candidates per prompt](figures/candidates_indian_man.png)

[indian_woman: all three candidates per prompt](figures/candidates_indian_woman.png)

[middle_eastern_man: all three candidates per prompt](figures/candidates_middle_eastern_man.png)

[middle_eastern_woman: all three candidates per prompt](figures/candidates_middle_eastern_woman.png)

[white_man: all three candidates per prompt](figures/candidates_white_man.png)

[white_woman: all three candidates per prompt](figures/candidates_white_woman.png)

## Prompt policy

The earlier Indian-image-specific coordinates have been replaced by one anatomical rule for every group. A baseline FaceLandmarker detection defines a box from the lower lip region to an extended chin region, with lateral face margins. Five positive points hypothesize lower-jaw/chin foreground; three negative points mark mouth and upper-cheek locations. The same baseline box/points are reused unchanged for the paired edits. Box-only and box-plus-points use the same box. No race/gender-specific thresholds, prompts, or presumed beard labels are used.

**Important:** positive anatomical points may land on skin or neck in a clean-shaven image. They are geometric hypotheses, not annotated beard pixels. Even box-only SAM can segment a non-beard object. A nonempty proposal is not beard detection, and the highest SAM quality score is not the probability of a beard.

## Baseline checks across all groups

![All man-prompt baselines](figures/baseline_overview_man.png)

![All woman-prompt baselines](figures/baseline_overview_woman.png)

Gender labels are prompt conditions, not assumptions about the presence or absence of facial hair.

## Paired images and masks by group

### black_man

![black_man originals and selected masks](figures/black_man.png)

### black_woman

![black_woman originals and selected masks](figures/black_woman.png)

### east_asian_man

![east_asian_man originals and selected masks](figures/east_asian_man.png)

### east_asian_woman

![east_asian_woman originals and selected masks](figures/east_asian_woman.png)

### hispanic_man

![hispanic_man originals and selected masks](figures/hispanic_man.png)

### hispanic_woman

![hispanic_woman originals and selected masks](figures/hispanic_woman.png)

### indian_man

![indian_man originals and selected masks](figures/indian_man.png)

### indian_woman

![indian_woman originals and selected masks](figures/indian_woman.png)

### middle_eastern_man

![middle_eastern_man originals and selected masks](figures/middle_eastern_man.png)

### middle_eastern_woman

![middle_eastern_woman originals and selected masks](figures/middle_eastern_woman.png)

### white_man

![white_man originals and selected masks](figures/white_man.png)

### white_woman

![white_woman originals and selected masks](figures/white_woman.png)

## Measurements and limits

Raw mask area, lower-face ROI overlap, lip overlap, candidate agreement and between-prompt IoU are saved for diagnosis. Paired changes use the same baseline ROI denominator. These are **SAM-proposal changes**, not accepted changes in beard coverage. The geometric ROI extends below the chin and may include skin/neck. It is not semantic ground truth.

The top candidate is selected by SAM quality score before looking at conditions; all alternatives remain available. High agreement or model confidence can be confidently wrong about what object was segmented. Failed detections/segmentations remain in status tables and are not silently excluded. There is one seed per group, so no group-level segmentation-accuracy or fairness claims are made.

Inspect whether the selected regions follow facial hair, especially in images without an obvious beard. If substantial non-hair regions are selected, do not score all4,200 images as beard area. SAM may remain useful as a prompt-assisted mask tool for visible beards, but would need a separate reliable presence gate or reviewed masks before quantitative semantic use.

## Preservation and reproduction

originals/: untouched512px PNG hardlinks. landmarks/: full landmarks, fixed soft ROI and lips polygon. config/frozen_prompts.json: all boxes, points and labels. masks/: every candidate and selected mask at512px. sam_arrays/: lossless candidate masks, quality scores and low-resolution logits. image_embeddings/: cached SAM embeddings. overlays/ and figures/: separate derivatives. logs/events.jsonl: append-only attempts. results/: per-candidate and selected-proposal diagnostics.

The selected diffusion/SAE image provenance is recorded with SHA-256 cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6. SAM ViT-B weights and the FaceLandmarker task are separately hashed. Frozen SAM inference runs in float32; no SAE or SDXL model is loaded.

To resume: submit instance-segmentation/all_group_sam/pilot.sbatch RUN_DIRECTORY. To rebuild this report: run report.py --run RUN_DIRECTORY in the recorded NumPy/Pillow/Matplotlib environment. Do not rerun prepare.py on an existing directory; it creates a new timestamped pilot.


## Publication scope

This GitHub directory contains derived previews, diagnostic CSVs, frozen code and configurations. Original images, all full-resolution masks, embeddings and logits remain in the permanent experiment directory:

`/n/fs/goose/experiments/mid0_sae_selectivity/sam_beard_all_groups_20261116_233008_010361`

Reproduction on the research host:

```bash
sbatch instance-segmentation/all_group_sam/pilot.sbatch /n/fs/goose/experiments/mid0_sae_selectivity/sam_beard_all_groups_20261116_233008_010361
```

The job resumes completed predictions from its journal. No new diffusion images are generated. `local_full_output_manifest.csv` describes the complete local archive, including files intentionally excluded from GitHub.
