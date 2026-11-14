# Beard-segmentation pilot status

The first SCHP ATR attempt used a generic hair class and missed much of the
visible beard. The prompt-guided SAM attempt also selected much of the face.
Neither mask is accepted as beard ground truth.

## What is available

- The 2023 beard-segmentation paper describes a BiSeNet trained with manually
  outlined facial-hair masks. Its linked ready-to-run checkpoint is the generic
  19-class face parser, not the paper's binary beard model.
- A 2026 facial-attribute study publishes beard-labeled annotations for FFHQ
  IDs, but the repository does not include the original FFHQ photos or a
  trained checkpoint.
- For these generated Feature 4228 images, this pilot uses hand-drawn masks as
  ground truth before any alpha-area comparison.

## Start the pilot

```bash
python instance-segmentation/prepare_beard_pilot.py
```

By default this prepares seed 30 at alphas -7, +1, and +7 for all 12 categories
in `/tmp/goose_beard_annotation_pilot/`. Face Landmarker cheek, jaw, and chin
keypoints define a lower-face crop for annotation. This crop narrows the task but
does not segment beard pixels; annotators must trace visible beard hair and leave
skin, lips, and the separate mustache unlabeled. Matching full images with the
crop boundary are saved in `context_images/`; the cropped images for LabelMe are
in `images/`. See `ANNOTATION_GUIDE.md` for the exact labeling steps. To include
more seeds, pass `--seeds 30 31 32`.

If the pilot already exists, regenerate it with `--overwrite`.

The current 36-image pilot is checked in at
[`results/feature_4228/beard_annotation_pilot/`](results/feature_4228/beard_annotation_pilot/).
To regenerate that repository copy, run from the repository root:

```bash
python instance-segmentation/prepare_beard_pilot.py \
  --out instance-segmentation/results/feature_4228/beard_annotation_pilot \
  --overwrite
```

Once annotations are saved, export binary masks with:

```bash
python instance-segmentation/export_labelme_beard_masks.py
```

The first pilot is 36 images. If its labels look consistent, the next step is a
small train/validation experiment using a beard-specific binary segmentation
model, followed by visual review on held-out categories and seeds.

References: [Beard Segmentation and Recognition Bias](https://arxiv.org/abs/2308.15740),
[2026 facial-hair segmentation dataset](https://github.com/IFrajtag/Facial_hair_segmentation_analysis/tree/main/data/dataset_1_training_ffhq).
