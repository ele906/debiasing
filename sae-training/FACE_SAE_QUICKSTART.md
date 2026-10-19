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

Each race/gender category gets a contact sheet under
`reconstructions/<sample_id>/contact_sheet.jpg`. Rows are epochs; columns are
the full SAE reconstruction followed by pixel heatmap overlays for the same
top three validation-ranked features. The original image is saved separately
as `original.png` in that category folder.

The overview puts men in the left panel and women in the right panel, with
prompt groups arranged by row. It is saved to
`sae-training/runs/face_sae_020/reconstructions/all_samples.jpg`.
If rendering stops partway through, rerun the command with `--resume` to reuse
completed sample sheets and finish the remaining ones.

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
