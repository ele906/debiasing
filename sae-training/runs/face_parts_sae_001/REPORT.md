# Shared four-region SAE fine-tuning

## What was trained

One pretrained Top-K SAE was fine-tuned on U-Net activation vectors inside
automatically detected eye, eyebrow, nose, and ear boxes. The SAE has a shared
linear encoder, Top-K selection, and shared linear decoder. The U-Net/image
generator was frozen. Region and race/gender category sampling were balanced
within each epoch. Epoch 20 was selected by the mean of the four
macro-category validation MSE values.

## Held-out test reconstruction

| Region | Images | Original SAE MSE | Fine-tuned SAE MSE | Change |
| --- | ---: | ---: | ---: | ---: |
| Eyes | 90 | 2.1007 | 1.4267 | -32.1% |
| Eyebrows | 28 | 2.1241 | 1.4559 | -31.5% |
| Nose | 13 | 2.2286 | 1.5092 | -32.3% |
| Ears | 18 | 2.1090 | 1.6102 | -23.7% |

Across regions, macro-category MSE decreased from 2.1306 to 1.5074 (29.3%).
The metric is SAE activation reconstruction error, not image reconstruction
quality.

## Candidate feature rankings

The first three validation-ranked features are eyes: 3701, 3593, 1600;
eyebrows: 3701, 1600, 1237; nose: 1664, 3593, 1991; ears: 4858, 3725,
2230. Features can rank for more than one region; these are spatial activation
associations, not dedicated semantic labels.

Epoch heatmap sheets use epochs 0, 4, 8, 12, 16, and 20 and are in
`region_heatmaps/<region>/feature_<id>.jpg`. The source annotations are
automatic and unreviewed. Test coverage is especially small for nose and ears,
so treat those estimates as preliminary and inspect the boxes in the sheets.

## Held-out per-image feature maps across training

The [complete held-out gallery](individual_part_epoch_grids/index.html) contains
one sheet for each of the 240 test portraits. Each sheet uses rows for face,
eyes, eyebrows, nose, and ears, and columns for epochs 0, 4, 8, 12, 16, and 20.
The cyan outline marks the detected face box; the yellow outline marks that
part's automatic box. A missing part box indicates missing usable annotation,
not that the person lacks that feature. The face row is shown within the face
box; ears are compared against the non-ear image area because ear boxes can
extend beyond the face box.

For each part, a single SAE feature was selected using only validation data:
the feature with the highest macro-category mean activation contrast between
the annotated part region and its reference region at epoch 20. The same
feature is visualized at each checkpoint, while the test images are held out
from feature selection. The resulting feature IDs are face 1664, eyes 3701,
eyebrows 1600, nose 1664, and ears 4858. These are candidate spatial
associations, not proof that a feature exclusively represents a semantic part.
The validation contrasts and coverage are in
[`validation_selected_features.csv`](individual_part_epoch_grids/validation_selected_features.csv).

## Reproducibility files

- `settings.json`: checkpoint hash, split/category counts, sampler, and run settings.
- `training_history.json`: training loss and validation MSE after each epoch.
- `heldout_reconstruction.json`: original-versus-fine-tuned validation and test metrics.
- `region_localized_features.csv`: validation-ranked feature contrasts by region.
- `checkpoints/`: pretrained epoch 0 and fine-tuned snapshots every four epochs.
