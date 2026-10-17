# Face-focused SAE pilot

## First pass: face-region SAE fine-tuning

The source set contains 1,200 generated portraits and saved U-Net residual activations at the same layer used by the original SAE. The original generator stays frozen; a face-specialized SAE would be initialized from the existing SAE and fine-tuned on face-region activations.

The C2C object detector labels a `human face` box in 1,200 images (1,199 unique samples). These boxes are not reviewed: the median box covers about 30% of the image, and the largest cover about 77%. This first pass uses those boxes only as spatial masks, so the result is exploratory and may include hair, shoulders, or background. No detector or box-prediction head is trained.

Run the fine-tune with:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python sae-training/fine_tune_face_sae.py
```

The run writes to `sae-training/runs/face_sae_001/`. It starts from the original SAE checkpoint, trains on the existing train split only, chooses an epoch using validation face-region reconstruction error, reports original-versus-fine-tuned reconstruction on validation and test, ranks features by validation face-versus-outside activation, and renders test heatmaps. The image generator and original checkpoint are left unchanged. Phrase grounding is not used.

`gallery.html` and `face_region_review.csv` remain available as an optional balanced audit set: 10 examples from each of the 12 prompt categories. Cyan outlines show the object detector's `human face` proposal. In the CSV, mark `box_valid` with `1`, `0`, or `uncertain`; enter corrected normalized coordinates when needed. Reviewing the boxes would support a stronger follow-up.

## Proposed training and evaluation

The first pass uses automatic boxes. A stronger follow-up can replace them with reviewed boxes. Compare original and fine-tuned SAEs on held-out portraits: reconstruction error inside and outside the face region, feature activation maps, and top-activating image patches. Inspect whether candidate features align with eyes, ears, hair, or other face details across prompt groups. A face box can focus the SAE on face-region activations, but it does not label eyes, ears, hair, or skin tone individually; those interpretations need separate region labels and visual checks. Do not claim causal image control based on heatmaps alone.

The box proposals came from an automatic detector, so any results remain provisional until the relevant regions are reviewed. Do not use detector boxes as verified ground truth.
