# Flower SAE image intervention

This folder collects the flower image-generation intervention results and comparison pictures. The SAE hook was applied at one SDXL Turbo U-Net layer, `unet.down_blocks.2.attentions.1`. In the four-step run, the hook ran at that same layer on each of the four denoising steps. This is not a multi-layer SAE experiment.

## Setup

The SAE was fine-tuned from the pretrained Top-K checkpoint with a C2C box-prediction head. Training used 720 images and a 720/240/240 train/validation/test split. The SAE and head were trained jointly for five epochs. The loss combined normalized reconstruction MSE, objectness binary cross-entropy, and box Smooth L1 regression, weighted 1, 1, and 5. The SAE learning rate was 1e-6 and the head learning rate was 1e-3. The frozen-SAE baseline used the same objective and updated only the head.

For the image intervention, features 940, 2810, and 1855 were selected using their contribution to the flower score from the trained box head on training images. Feature 3459 was the activity-matched random control. The intervention scaled selected feature contributions while preserving the SAE reconstruction error.

## One-step result

We replayed 20 held-out East Asian woman prompts: 12 C2C flower-positive and 8 negative. Source replays were pixel-identical. Zeroing the selected features changed the images by mean RGB MAE 11.03/255 on positives and 10.99/255 on negatives; half attenuation gave 6.10 and 6.38. The random-feature control changed them by 1.44 and 1.64. In inspected positive images, the flower ornament remained visible. Changes were similarly large on positives and negatives, and affected other image details.

![One-step C2C-positive comparison: source, replay, SAE feature ablation, no-op, and random control](images/one_step_positive_seed007.jpg)

![One-step C2C-negative comparison](images/one_step_negative_seed086.jpg)

## Four-step result

We repeated the intervention on four positive and four negative prompts using four inference steps. Feature ablation changed images by mean RGB MAE 32.63/255 on positives and 25.16/255 on negatives; half attenuation gave 20.86 and 18.54. The random control averaged 6.84/255. The flower remained visible in the inspected positive example.

The four-step baseline already differed substantially from the original one-step source images: mean source-to-replay MAE was 66.64/255 for positives and 57.99/255 for negatives. The larger intervention effect came with substantial image drift, so four steps did not solve flower removal in this setup.

![Four-step C2C-positive comparison: original one-step source, four-step replay, SAE feature ablation, no-op, and random control](images/four_step_positive_seed007.jpg)

## Frozen versus fine-tuned feature comparison

On a matched held-out set of 20 East Asian woman prompts (12 positive, 8 negative), the trained box head's flower AUC was 0.792 with the fine-tuned SAE and 0.729 with the frozen SAE. Average precision was 0.892 and 0.844, respectively. But individual SAE feature activations did not become more reliable flower indicators: the median held-out AUC among the top ten features selected on training data was 0.531 for fine-tuned and 0.578 for frozen. These small results suggest the head learned a more useful combination of spatial codes, not a consistently stronger standalone flower feature.

All labels and boxes here are C2C working targets and were not manually audited. Pixel MAE measures overall image change, not whether a flower disappeared. The current experiment did not establish reliable flower removal.

## Detailed artifacts

- [Full results, training details, and comparison images](https://github.com/ele906/sae-training/blob/main/RESULTS.md)
- [Feature AUC comparison data](https://github.com/ele906/sae-training/tree/main/visualizations/flower_auc_frozen_vs_finetuned_001)
- [One-step per-image metrics](https://github.com/ele906/sae-training/blob/main/visualizations/flower_intervention_001/metrics.csv)
- [Four-step per-image metrics](https://github.com/ele906/sae-training/blob/main/visualizations/flower_intervention_4step_001/metrics.csv)
- [Training implementation](https://github.com/ele906/sae-training/blob/main/train_box_sae.py)
- [Intervention runner in this folder](ablate_flower_features.py)

The runner expects the local `sae-training` workspace, cached model weights, residuals, and generated source images to be available alongside this folder. Those large/private run artifacts are not included here.
