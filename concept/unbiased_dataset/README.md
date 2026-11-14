# C2C-style extraction for the unbiased image collection

This folder contains a separate Florence-2 extraction run for
`data/datasets/unbiased/images`. It is kept separate from the existing
`concept/` pilot and full-dataset outputs. The collection has 600 images: 50
per race/gender prompt category across 12 categories.

`prepare_input.py` validates the category balance and writes `input.csv` with
absolute image paths. All rows use `split=unlabeled`: this is a descriptive
detector audit, not a train/validation/test experiment. The extractor writes
per-image captions, phrase-grounding boxes, object-detection boxes, and
visualized grounding outputs under `results/extracted/`; summary CSVs and a
report go under `results/`. A postprocessing step creates
`results/category_analysis/bounding_boxes.csv`, box visualizations, and
category-level detection counts.

Run from `/n/fs/goose` on the cluster:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  concept/unbiased_dataset/prepare_input.py
sbatch concept/unbiased_dataset/extract.slurm
```

The Slurm job uses the same pinned, cached Florence-2 checkpoint and isolated
dependencies as the existing C2C-style extraction. Predictions are noisy
automatic annotations, not verified labels. Review images and boxes before
using them for feature selection or training. In particular, small regions
such as noses may be omitted by object detection; phrase grounding can return
different coverage and should also be reviewed.
