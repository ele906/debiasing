# Beard segmentation feasibility probe

This experiment checks whether the SCHP human-parsing model's generic `hair`
class captures beard pixels in Feature 4228 images. The ATR model predicts
classes such as `hair`, `face`, `hat`, and `scarf`; it has no dedicated beard
class. A jaw/chin ROI from Face Landmarker is used only to focus the review on
the lower face. The resulting intersection is therefore a *candidate beard
mask*, not a validated beard segmentation.

The first run uses two Indian-man seeds at alphas `-7`, `+1`, and `+7`. Review
the saved overlays before deciding whether the predicted hair class follows
the beard boundary well enough to compute beard-area statistics. The model
may label facial hair as `face`, omit it, or mix it with scalp hair.

## Run

Install the Python dependencies in `requirements.txt`, then run. The probe uses
the model's quantized ONNX checkpoint for CPU inference and does not require
PyTorch or Transformers:

```bash
python instance-segmentation/probe_schp_beard.py
```

The first run downloads the 66 MB SCHP ATR ONNX checkpoint from Hugging Face. To try a
different category, seeds, or alpha values:

```bash
python instance-segmentation/probe_schp_beard.py \
  --category middle_eastern_man --seeds 30 31 --alphas -7 1 7
```

Outputs are written under `instance-segmentation/results/feature_4228/beard_probe/`:

- `*_class_ids.png`: predicted semantic class ID per pixel.
- `*_hair_class_mask.png`: all pixels predicted as generic hair.
- `*_lower_face_roi.png`: Face-Landmarker jaw/chin region.
- `*_hair_in_lower_face_candidate.png`: intersection proposed for visual review.
- `*_candidate_overlay.png`: candidate mask over the source image.
- `probe_metrics.csv`: raw candidate pixel counts and lower-face proportions.

## Interpretation

The ATR label set includes generic hair and clothing classes, but no beard
class. A model trained on that label set cannot establish beard identity by
itself. The saved images are the key feasibility check: if the lower-face
intersection visibly traces beard pixels across examples, it may be usable as
a cautiously named proxy. Otherwise, use manual beard masks or a model trained
and validated specifically for facial hair. Do not interpret these candidate
pixel counts as beard area until that visual validation succeeds.

## First probe result

On the six Indian-man images (seeds 30 and 31 at alphas `-7`, `+1`, and `+7`),
the model missed much of the visible white facial hair and marked some cheek
pixels. It is not reliable enough for beard-area comparisons. See the [sample
overlays and counts](results/feature_4228/beard_probe/README.md).

## References

- Liang et al., [Human Parsing with Contextualized Convolutional Neural Network](https://openaccess.thecvf.com/content_iccv_2015/papers/Liang_Human_Parsing_With_ICCV_2015_paper.pdf), describing the ATR classes including face, hat, scarf, and hair.
- [SCHP ATR model package and label map](https://github.com/pirocheto/schp-human-parsing).
- Li et al., [Self-Correction for Human Parsing](https://arxiv.org/abs/1910.09777).
