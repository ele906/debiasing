# Supplemental held-out test set

Generated 1,200 additional portraits with the same prompt template as the
original dataset, using seeds 100–199 (100 images in each of 12 prompt
categories). All images are marked `test`; none were used for training or
checkpoint selection. Florence-2 produced automatic, unreviewed boxes. Two
images did not receive a usable face box.

## Usable region boxes and held-out MSE

| Region | Usable images | Prompt categories | Original SAE MSE | Fine-tuned SAE MSE | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eyes | 497 | 12 | 2.0879 | 1.4276 | 31.6% |
| Eyebrows | 156 | 11 | 2.1168 | 1.4961 | 29.3% |
| Nose | 73 | 9 | 2.1524 | 1.5020 | 30.2% |
| Ears | 77 | 9 | 2.1115 | 1.6238 | 23.1% |

The mean across region macro-category MSEs decreased from 2.1192 to 1.5411
(27.3%). This is SAE activation reconstruction error, not photo quality.
Images without a usable region box are excluded from that region's MSE; the
usable-image counts above make that detector coverage explicit.

## Files

- `manifest.csv`: the new test-only samples, seeds, and prompt categories.
- `annotations/bounding_boxes.csv`: Florence-2 phrase/object boxes.
- `face_parts_evaluation.json`: full pooled and per-category baseline/tuned metrics.
- `region_heatmaps/`: pixelated feature overlays at epochs 0, 4, 8, 12, 16, and 20.
- `detector/REPORT.md`: detector phrase counts and annotation notes.

The test images use the original simple prompts. Some portraits still have
head coverings, unusual poses, or obscured facial regions, and automatic boxes
can be wrong. Inspect the overlays and boxes before treating the metrics as
ground-truth region measurements.
