# SAE Box Prediction and Flower Feature Intervention

## Goal

This work tested two linked ideas: whether an SAE with a box-prediction head could imitate Concept2Concept (C2C) boxes for flowers and kimonos, and whether SAE features selected by that head could be suppressed to remove a flower from generated images.

The diffusion generator was not fine-tuned. C2C detections and boxes are automatic labels, so agreement with them is not independent proof of accurate object detection.

## Part 1: C2C box prediction

We replayed 1,200 existing SDXL portraits and captured spatial residuals at `unet.down_blocks.2.attentions.1`. Each residual grid was 16 × 16 with 1,280 channels. The fixed split was 720 training, 240 validation, and 240 test images. We trained a spatial box head with the pretrained SAE frozen, then trained the same setup with the SAE fine-tuned jointly for five epochs. Validation loss selected the fine-tuned checkpoint; the reported results are on the held-out test split.

| Concept | SAE setting | C2C positives / 240 | Objectness AP | Mean IoU on positives | Recall at score ≥0.5 and IoU ≥0.5 |
|---|---|---:|---:|---:|---:|
| Kimono | Frozen | 21 | 0.932 | 0.575 | 0.524 |
| Kimono | Fine-tuned | 21 | 0.921 | 0.550 | 0.476 |
| Flower | Frozen | 13 | 0.799 | 0.035 | 0.000 |
| Flower | Fine-tuned | 13 | 0.840 | 0.087 | 0.000 |

Fine-tuning did not improve kimono localization. Flower objectness ranking improved slightly, but localization remained poor: mean IoU was 0.087 and no prediction reached the IoU threshold for recall. The flower test set had only 13 C2C-positive images. These metrics measure imitation of C2C's unreviewed annotations, not independently verified boxes.

## Part 2: Flower feature intervention

Using the fine-tuned SAE and box head, we selected three SAE features (940, 2810, and 1855) from training images based on their contribution to the flower score. We chose feature 3459 as a roughly activation-matched random control. We then suppressed the three candidate features **globally at the selected U-Net block** during replay of 20 held-out East Asian woman images: 12 labeled flower-positive and 8 labeled flower-negative. This intervention was global; the predicted boxes were used to train the box head and select features, not as a spatial mask for this edit.

| Held-out C2C label | Images | Three flower features, strength 0 MAE | Same features, strength 0.5 MAE | Random feature, strength 0 MAE |
|---|---:|---:|---:|---:|
| Positive | 12 | 11.027 | 6.097 | 1.441 |
| Negative | 8 | 10.993 | 6.377 | 1.644 |

Strength 1 was an exact no-op, and all source replays matched their originals exactly. Suppressing the three candidate features changed pixels much more than the random feature, but the change was nearly as large on C2C-negative images as on positives. Visual inspection did not show clear flower removal. This is pixel change, not an object-presence score; the candidate edit also changed other parts of some images.

## Conclusion

The box head learned some agreement with C2C's labels, but flower boxes were poorly localized. Features selected from the box head caused substantial image changes when suppressed, yet the flowers remained visible and similar changes occurred on images labeled flower-negative. So this run did not establish reliable flower control. The next step for a stronger claim is human-verified flower labels and boxes, followed by direct review or an independently validated measure of flower presence and collateral image changes.

## Files

- [Box prediction results](../../sae-training/RESULTS.md)
- [Training and experiment notes](../../sae-training/README.md)
- [Flower intervention runner](../../sae-training/ablate_flower_features.py)
- [Flower intervention settings](../../sae-training/runs/flower_intervention_001/settings.json)
- Per-image intervention measurements: `sae-training/runs/flower_intervention_001/metrics.csv` (local run output)
- Visual comparisons: `sae-training/runs/flower_intervention_001/all_test_comparisons_page_*.jpg` (local run output)

## Probe follow-up

Human review and the probe experiment are still pending. Once the image labels are reviewed, prepare separate Markdown documentation for the probe setup and results before presenting it.
