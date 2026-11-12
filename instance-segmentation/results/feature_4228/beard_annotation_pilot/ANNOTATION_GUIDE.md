# Feature 4228 beard-labeling pilot

Prepared 36 images (12 categories, 1 seed(s), alpha values -7, +1, +7).

## Labeling rule

The annotation images are crops bounded from Face Landmarker cheek/jaw/chin keypoints. They show the lower face only to make beard labeling easier; the crop is not a beard segmentation. Use the matching full image in `context_images/` if you need context. Draw polygons only around visible beard hair on cheeks, jaw, and chin. Exclude skin, lips, scalp hair, clothing, and the separate mustache. Do not outline the face or fill the entire crop. If no beard is visible, save a reviewed annotation with no beard polygon. Use the same boundary rule at every alpha.

## LabelMe

Install LabelMe in the environment where you can use a graphical desktop, then run from the repository root:

```bash
python -m pip install labelme
```

Open the copied pilot images with:

```bash
labelme --nodata --labels instance-segmentation/results/feature_4228/beard_annotation_pilot/labels.txt instance-segmentation/results/feature_4228/beard_annotation_pilot/images
```

Save polygon shapes with the exact label `beard`; JSON files should be saved beside their matching images. After labeling, export binary masks with:

```bash
python instance-segmentation/export_labelme_beard_masks.py --pilot instance-segmentation/results/feature_4228/beard_annotation_pilot
```

These hand-labeled masks will be ground truth for a small transfer-learning pilot. The failed generic-hair and first SAM masks are not used as labels.
