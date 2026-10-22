# Independent face-parts test set (seeds 200–299)

## Dataset and protocol

This is a new, test-only set of 1,200 generated portraits: 100 images for each
of the same 12 race/gender prompt categories, using seeds 200–299. The seed
range is disjoint from both the training data and the earlier supplemental
test set (seeds 100–199). No SAE training, checkpoint selection, or feature
ranking was performed on these images.

Florence-2 automatically produced region boxes. The annotations have not been
reviewed by a person. The fixed pretrained SAE and the fixed face-parts
fine-tuned SAE were evaluated on residual activations from these images. The
feature heatmaps use features ranked on the training run's validation data.

## Held-out activation reconstruction

MSE below is measured on SAE activation vectors inside automatically annotated
regions; it is not image-pixel error. Values are macro-averaged across prompt
categories with usable annotations.

| Region | Images with usable box | Prompt categories | Pretrained SAE MSE | Fine-tuned SAE MSE | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eyes | 477 | 12 | 2.0825 | 1.4395 | 30.9% |
| Eyebrows | 162 | 11 | 2.0837 | 1.4732 | 29.3% |
| Nose | 85 | 8 | 2.0988 | 1.4508 | 30.9% |
| Ears | 77 | 9 | 2.1629 | 1.7149 | 20.7% |
| Mean across regions | — | — | 2.1070 | 1.5196 | 27.9% |

There is a usable face box for 1,199 of the 1,200 images. A usable region box
requires enough annotated activation cells for evaluation. The lower MSE means
the fine-tuned SAE reconstructed these held-out activations more closely. It
does not show that the SAE detects or semantically represents facial parts.

## Files

- `face_parts_evaluation.json`: full pooled and per-category metrics.
- `manifest.csv`: prompt groups, seeds, test split, and relative image paths.
- `annotations/bounding_boxes.csv`: aggregated automatic region boxes.
- `contact_sheets/`: plain generated portraits grouped by prompt category.
  These are for browsing the images; they do not show training epochs.
- `final_feature_localization/`: feature-to-region maps from the selected SAE
  checkpoint only. One sheet per candidate feature; rows show categories with
  the full-image activation overlay, face box, and target-part box. No epoch
  columns appear here.
- `region_heatmaps/<region>/feature_<id>.jpg`: epoch comparison sheets. Each
  row is a prompt category; columns show epochs 0, 4, 8, 12, 16, and 20 for the
  same held-out portrait. Each tile has the region's automatic bounding box
  and a pixelated jet-style overlay for that feature. For example, see
  [`region_heatmaps/eyes/feature_1600.jpg`](region_heatmaps/eyes/feature_1600.jpg).
- `detector/REPORT.md`: detector summary and coverage details.
- `generation_settings.json`: generation model, prompts, seeds, and settings.

The history contains metrics for all 20 epochs, but visual checkpoints were
saved every four epochs. Therefore, the feature sheets show six snapshots
(including the pretrained epoch 0), not all 21 epochs. To produce overlays for
every epoch, the SAE must be trained again with `--snapshot-interval 1` before
rendering the sheets.

For the final-checkpoint maps without epoch comparisons, start at
[`final_feature_localization/index.md`](final_feature_localization/index.md).

Individual generated images, cached model features, and residual activations
are stored locally and are not included in the GitHub repository.
