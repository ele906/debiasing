# SAE Box Prediction and Flower Feature Intervention

## What I tested

I ran two related tests: first, could an SAE-based model learn to predict Concept2Concept (C2C) boxes for flowers and kimonos? Second, could features selected using that model be suppressed to remove a flower from a generated image?

I did not fine-tune the diffusion generator. C2C supplied the labels and boxes, so agreement with C2C does not show that the predictions are independently correct.

## Part 1: C2C box prediction

I replayed 1,200 existing SDXL portraits and saved spatial residuals from `unet.down_blocks.2.attentions.1`. Each residual grid was 16 × 16 × 1,280. I kept the 720/240/240 train, validation, and test split. I trained the same spatial box head twice: once with the pretrained SAE frozen, and once with the SAE fine-tuned jointly for five epochs. I chose the checkpoint by validation loss and report results on the held-out test set.

| Concept | SAE setting | C2C positives / 240 | Objectness AP | Mean IoU on positives | Recall at score ≥0.5 and IoU ≥0.5 |
|---|---|---:|---:|---:|---:|
| Kimono | Frozen | 21 | 0.932 | 0.575 | 0.524 |
| Kimono | Fine-tuned | 21 | 0.921 | 0.550 | 0.476 |
| Flower | Frozen | 13 | 0.799 | 0.035 | 0.000 |
| Flower | Fine-tuned | 13 | 0.840 | 0.087 | 0.000 |

Fine-tuning did not improve the kimono boxes. For flowers, the objectness score improved a little, but the boxes were still poor: mean IoU was 0.087, and none met the IoU threshold for recall. There were only 13 C2C-positive flower images in the test set. These scores measure how well the model copied C2C's unreviewed labels, not whether the boxes were right.

## Part 2: Flower feature intervention

I used the fine-tuned SAE and box head to select three features (940, 2810, and 1855) based on how they contributed to the flower score on training images. Feature 3459 was the roughly activation-matched random control. Then I suppressed the three candidates **across the whole image** at the selected U-Net block, replaying 20 held-out East Asian woman images: 12 C2C positives and 8 negatives. I did not use the predicted boxes as a mask for this intervention; I used the box model to train the features and help select them.

| Held-out C2C label | Images | Three flower features, strength 0 MAE | Same features, strength 0.5 MAE | Random feature, strength 0 MAE |
|---|---:|---:|---:|---:|
| Positive | 12 | 11.027 | 6.097 | 1.441 |
| Negative | 8 | 10.993 | 6.377 | 1.644 |

Strength 1 was an exact no-op, and every source replay matched its original. Suppressing the candidate features changed more pixels than suppressing the random feature, but the change was just as large on C2C negatives as on positives. I did not see the flowers disappear. The edits changed other parts of some images too. Pixel change alone does not measure flower removal.

## Conclusion

The box head learned to copy some of C2C's labels, but it did a poor job locating flowers. The selected features changed the images when suppressed, but the flowers stayed visible, and the edits changed negative examples just as much. I did not find reliable flower control in this run. I’d need human-checked labels and boxes, then a direct measure of flower presence and changes elsewhere in the image.

## Files

The box-prediction numbers and flower-intervention results are summarized above. I have more detailed run files and flower comparison grids locally, but they are not in the GitHub repo yet.

## Probe follow-up

I have not done the probe experiment yet. Once I'm ready to work on it, I need to review the labels and write up the probe setup and results separately.
