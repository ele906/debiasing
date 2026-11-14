# Face Landmarker analysis

## What Face Landmarker did here

MediaPipe Face Landmarker finds face landmarks: points describing facial
geometry. We used those points to draw a soft region above the forehead and
outside the temples, down to the eyebrow line. For each alpha and seed, we
measured image changes inside that region, then averaged the changed-pixel
maps and areas across seeds.

The matrix has one row per prompt category and one column per alpha. Each cell
is mean changed area ± standard deviation across seeds. The colored images
show where changes recur. This measures change in a landmark-anchored region;
it does not detect or measure hijab, hats, hair, or other objects. Earlier
lower-face polygons were also geometry-based and should not be interpreted as
beard segmentation.

## Corrected reports

- [Feature 2023, first three groups](results/feature_2023/three_groups/README.md)
- [Feature 2023, other nine groups and all-category matrix](results/feature_2023/other_nine_groups/README.md)
- [Feature 1650, all categories and matrix](results/feature_1650/head_covering_region_report/README.md)
- [Feature 4228, all categories and matrix](results/feature_4228/head_covering_region_report/README.md)

## Reusable scripts

- `render_feature_2023_upper_face_groups.py` runs the landmark-region analysis
  on a feature's existing alpha-sweep images. The feature path is configurable
  with `--run`; categories can be selected with `--groups`.
- `plot_feature_2023_head_region_alpha.py` makes either the alpha trend chart
  or category-by-alpha matrix from the report CSVs (`--matrix`).
- `render_full_image_change_report.py` makes an unconstrained whole-image
  change report without landmarks.
- `render_landmarked_lower_face_change_report.py` is an earlier exploratory
  lower-face geometry analysis; it is not a beard detector.

The local MediaPipe package and task model are in `.deps/` and `models/`.
