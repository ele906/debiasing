# Landmark-constrained feature-change reports

`render_landmarked_lower_face_change_report.py` turns an alpha sweep into a
lower-face / extended-chin change report. It uses MediaPipe Face Landmarker to
build a smooth per-seed measurement loop, then reports how frequently pixels
inside that loop change relative to the seed baseline.

It produces, for every category:

- a category-average baseline portrait with the cyan landmark ROI loop;
- a blue → green → yellow → red occurrence grid for every alpha;
- a blurred mean RGB-change (attenuation) grid for every alpha;
- tidy CSV metrics and a cross-category Markdown table.

The generated score is **landmark-constrained lower-face change**, not a
semantic beard segmentation. This keeps the result defensible until a beard
segmenter is deliberately added.

## Run Feature 4228

The current local setup is in `.deps/` and `models/` (both ignored by Git):

```bash
python face-landmarker/render_landmarked_lower_face_change_report.py
```

The completed Feature 4228 output is at:

`layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_4228/beard_landmarker_report/`

To use another alpha-sweep run:

```bash
python face-landmarker/render_landmarked_lower_face_change_report.py \
  --run /path/to/feature_run \
  --out /path/to/output_report
```

Use `--change-threshold` to tune the per-pixel RGB-MAE threshold (default:
12 on the 0–255 scale). The default model is the official MediaPipe Face
Landmarker task file in `models/face_landmarker.task`.
