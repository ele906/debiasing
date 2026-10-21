# Face SAE quickstart

These commands use the `reno` Python environment and the completed run at
`sae-training/runs/face_sae_020`. Run them from `/n/fs/goose`.

## Plot the training metrics

The trainer saves epoch metrics in `training_history.json` and held-out metrics
in `heldout_reconstruction.json`. Run the plotting script to make a PNG and a
CSV copy of the epoch metrics:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/plot_face_sae_run.py \
  sae-training/runs/face_sae_020
```

The script writes these files into the run folder:

- `training_and_test_metrics.png` — training loss, validation reconstruction
  error, and original-versus-fine-tuned test error.
- `training_history.csv` — one row per epoch, with `epoch`,
  `train_loss_mean`, `face_mse`, `outside_mse`, and `all_mse` columns.
- `REPORT.md` — a short summary of the run and its limitations.

To put the plot and CSV in a separate folder, add an output path:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/plot_face_sae_run.py \
  sae-training/runs/face_sae_020 \
  --output sae-training/runs/face_sae_020/my_plots
```

You can read the CSV in your own Python plotting code or open it in a
spreadsheet. The trainer's original metric files remain in the run folder.

## Regenerate pixelated feature heatmaps

The overlays use a blue-to-red `jet` palette. Nearest-neighbor scaling keeps
the 16×16 activation cells visibly pixelated. This command uses the saved SAE
and feature ranking; it does not train again:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_face_heatmaps.py \
  sae-training/runs/face_sae_020
```

The contact sheet is written to
`sae-training/runs/face_sae_020/feature_maps_jet/heatmaps/heldout_face_feature_maps.jpg`.
The per-image sheets are saved in the same `heatmaps` folder. Existing heatmaps
are preserved.

## Compare generated images across epochs

The trainer saves SAE checkpoints at epochs 0, 4, 8, 12, 16, and 20. Epoch 0
is the pretrained SAE before fine-tuning. To render the same held-out prompts
and seeds through each checkpoint:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_epoch_reconstructions.py \
  sae-training/runs/face_sae_020
```

Each race/gender category gets an individual comparison sheet under
`reconstructions/<sample_id>/contact_sheet.jpg`. For an across-category
comparison, the script also writes `reconstructions/feature_sheets/`:

- `feature_1664.jpg`, `feature_1600.jpg`, and `feature_1237.jpg` each show one
  feature across all categories. Rows are the 12 race/gender categories (Black
  Man, Black Woman, White Man, White Woman, and so on); columns are epochs
  0, 4, 8, 12, 16, and 20.
- `reconstruction_by_category.jpg` uses the same 12 category rows and epoch
  columns to compare full SAE reconstructions.

The feature IDs come from the run's validation-ranked feature CSV. Each
category row uses one held-out test seed, shown beside the category label.

The overview puts men in the left panel and women in the right panel, with
prompt groups arranged by row. It is saved to
`sae-training/runs/face_sae_020/reconstructions/all_samples.jpg`.
If rendering stops partway through, rerun the command with `--resume` to reuse
completed sample sheets and finish the remaining ones.

## Compare feature activations inside face parts

The box annotations include face, eye, eyebrow, nose, mouth/lip, and ear
regions for some held-out images. Generate region-specific heatmaps from the
saved residuals and epoch checkpoints with:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_face_region_heatmaps.py \
  sae-training/runs/face_sae_020
```

The script writes one sheet per region and top-ranked feature to
`sae-training/runs/face_sae_020/region_heatmaps/<region>/feature_<id>.jpg`.
Rows are race/gender categories and columns are checkpoint epochs. It selects
one held-out image with a matching region box for each category when one is
available; missing annotations are labeled. Ear matching excludes earrings.
These are exploratory overlays from the existing SAE features, not evidence
that a feature has learned a semantic face-part label. The part boxes are
automatic and should be reviewed visually.

Each overlay is scaled within its annotated region to show the activation
pattern. For activation magnitude comparisons, use
`region_heatmaps/regional_activation_summary.csv`; it records mean/max
activation and the fraction of active cells for each feature, epoch, region,
and selected image. Add `--resume` to replace sheets in an existing output
folder.

## Run a separate eyes-only fine-tuning experiment

This starts from the original pretrained SAE and trains only on activation
cells inside eye boxes. It balances sampling across the 12 race/gender groups,
chooses the checkpoint by macro-average validation eye reconstruction error,
and also measures whole-face reconstruction to check for loss outside the
eyes. It does not change `face_sae_020`.

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/fine_tune_eye_sae.py \
  --epochs 20 \
  --snapshot-interval 4 \
  --output sae-training/runs/eye_sae_001
```

The eye-box filter excludes boxes covering more than 12% of the image and
images with fewer than two 16×16 activation cells inside the boxes. The run
folder contains checkpoints, `training_history.json`,
`heldout_reconstruction.json`, `heldout_metrics.csv`,
`eye_localized_features.csv`, and `settings.json`.
These are exploratory results from automatic, unreviewed boxes.

To render part-specific overlays using the eye-fine-tuned SAE, run:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_face_region_heatmaps.py \
  sae-training/runs/eye_sae_001
```

When the run folder contains `eye_localized_features.csv`, the renderer uses
that ranking for its three feature sheets.

## Fine-tune one SAE on eyes, eyebrows, nose, and ears

This experiment starts from the same pretrained Top-K SAE and updates one
shared encoder and decoder. Each epoch samples equally from the four regions,
then balances race/gender categories among the available boxes for each region.
The training, validation, and test splits stay separate. Validation chooses
the checkpoint; test metrics are computed after selection.

Submit the GPU job from `/n/fs/goose`:

```bash
sbatch sae-training/train_face_parts_sae.sbatch
```

The job writes checkpoints at epochs 0, 4, 8, 12, 16, and 20 to
`sae-training/runs/face_parts_sae_001/checkpoints/`. The selected SAE,
per-region validation rankings, epoch history, held-out reconstruction
metrics, and exact data counts are saved in the same run folder. Its
`region_localized_features.csv` ranks features separately for each region.
Render one pixelated `jet` heatmap sheet per region and feature with:

```bash
sbatch sae-training/render_face_parts_sae.sbatch
```

The renderer uses each region's own validation-ranked features and the same
held-out test images across checkpoints. Output is in
`sae-training/runs/face_parts_sae_001/region_heatmaps/`.

Because nose and ear detections are sparse and all these labels are automatic,
read the per-region sample counts and inspect the boxes and overlays before
interpreting a feature as associated with a facial part.

## Expand the held-out face-part test set

To test the already-trained SAE on more images, generate 100 additional seeds
per race/gender prompt category (1,200 images). This reuses the original prompt
wording, uses disjoint seeds 100–199, and marks every new image as test-only.
The training and validation datasets and the selected checkpoint are untouched.
Submit generation, then queue Florence annotation, held-out metrics, and
heatmaps to run automatically after generation succeeds:

```bash
generation_job=$(sbatch --parsable sae-training/generate_face_parts_test_extension.sbatch)
sbatch --dependency=afterok:${generation_job} \
  sae-training/extract_face_parts_test_boxes.sbatch
```

Results are saved under
`sae-training/runs/face_parts_sae_001/test_extension/`. `face_parts_evaluation.json`
compares the original and fine-tuned SAE on the same newly generated images;
`region_heatmaps/` shows the selected features by epoch. Metrics only include
samples with usable automatic boxes. The JSON also reports the number of usable
images per region, so detection coverage is visible alongside reconstruction
error. The boxes remain unreviewed detector outputs.

For the completed 1,200-image extension, the region-specific evaluation report
is `sae-training/runs/face_parts_sae_001/test_extension/REPORT.md`.

## Validate eye, eyebrow, nose, and ear feature candidates

This analysis keeps feature selection and evaluation separate. The three eye
features come from the eye run's validation ranking. Eyebrow, nose, and ear
candidates are selected on validation activation contrast, then all candidates
are evaluated on held-out test images. It reports the average activation
difference between the annotated region and a reference area, with a
category-stratified image-bootstrap confidence interval. Eyes, eyebrows, and
nose use the rest of the detected face as the reference; ears use the rest of
the image because ear boxes can extend beyond the face detector box.

Run it from `/n/fs/goose`:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/validate_face_part_features.py \
  sae-training/runs/eye_sae_001
```

Results are saved to
`sae-training/runs/eye_sae_001/part_feature_validation/`. To replace existing
results after changing the analysis, add `--resume`. The CSV includes each
feature's test contrast, 95% interval, positive-image fraction, and sample
counts; the folder's `REPORT.md` summarizes the current results. Positive
contrast means the feature activates more inside that annotation than in its
reference area. It supports spatial association with these automatic boxes,
not proof that a feature semantically represents eyes, eyebrows, nose, or ears.

To make epoch sheets using different features for each region, run:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_region_specific_heatmaps.py \
  sae-training/runs/eye_sae_001
```

The script uses each region's validation-ranked candidates and renders them
only inside that region's box, on held-out test images. Each sheet keeps the
12 race/gender categories as rows and epochs 0, 4, 8, 12, 16, and 20 as
columns. Files are saved under
`sae-training/runs/eye_sae_001/region_specific_heatmaps/<region>/`.

## Start a new training run

Each run needs a new output directory. The existing `face_sae_020` folder
contains the completed 20-epoch run, so choose a different name for another
training job:

```bash
/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/fine_tune_face_sae.py \
  --epochs 20 \
  --snapshot-interval 4 \
  --output sae-training/runs/face_sae_021
```

Training requires CUDA. Image rendering uses the cached SDXL model by default;
the scripts accept `--allow-download` if the model is not cached and downloads
are available.
