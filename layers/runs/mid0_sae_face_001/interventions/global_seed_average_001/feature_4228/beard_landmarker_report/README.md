# Feature 4228 — landmark-constrained lower-face change report

This report measures intervention change *within* a MediaPipe Face Landmarker lower-face/extended-chin region. It is not a semantic beard segmenter: an included pixel is counted when its RGB MAE from that seed's baseline is at least **12.0** (0–255 scale). The jet overlay shows the fraction of seeds for which each ROI pixel crossed that threshold.

The cyan loop is the landmark-constrained measurement region. It extends below landmark 152 (chin) so long beards are not clipped. The `alpha_blurred_mean_pixel_change.png` panel is the requested attenuation visual: seed-level RGB-MAE maps are blurred before averaging.

## Mean changed lower-face area by alpha

Each cell is mean percent of the soft landmark ROI that changed, followed by its across-seed standard deviation. `+1` is the unmodified reconstruction in this run and should be near zero.

| Category | -7 | -5 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +5 | +7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Black Man | 10.0% ± 7.5 | 7.6% ± 6.1 | 5.0% ± 4.5 | 3.5% ± 3.6 | 2.1% ± 2.4 | 0.8% ± 1.2 | 0.0% ± 0.0 | 0.9% ± 1.3 | 2.5% ± 2.9 | 6.6% ± 5.5 | 11.2% ± 7.6 |
| Black Woman | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| East Asian Man | 0.6% ± 1.3 | 0.4% ± 0.9 | 0.2% ± 0.5 | 0.1% ± 0.4 | 0.1% ± 0.2 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.1% ± 0.2 | 0.3% ± 0.7 | 0.6% ± 1.3 |
| East Asian Woman | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Hispanic Man | 14.8% ± 13.5 | 11.5% ± 11.5 | 7.5% ± 8.5 | 5.3% ± 6.7 | 3.2% ± 4.7 | 1.2% ± 2.2 | 0.0% ± 0.0 | 1.2% ± 2.0 | 3.2% ± 3.9 | 8.2% ± 6.9 | 13.2% ± 9.6 |
| Hispanic Woman | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| Indian Man | 43.0% ± 15.8 | 39.3% ± 15.4 | 33.5% ± 14.6 | 28.6% ± 13.2 | 21.9% ± 11.1 | 12.3% ± 7.4 | 0.0% ± 0.0 | 13.4% ± 7.9 | 25.1% ± 11.6 | 41.2% ± 15.3 | 50.9% ± 16.3 |
| Indian Woman | 0.8% ± 1.6 | 0.6% ± 1.2 | 0.4% ± 0.8 | 0.3% ± 0.6 | 0.2% ± 0.4 | 0.1% ± 0.2 | 0.0% ± 0.0 | 0.1% ± 0.2 | 0.2% ± 0.4 | 0.4% ± 0.9 | 0.7% ± 1.3 |
| Middle Eastern Man | 51.7% ± 14.5 | 47.5% ± 14.3 | 40.1% ± 12.9 | 33.6% ± 11.6 | 24.6% ± 9.4 | 12.5% ± 5.7 | 0.0% ± 0.0 | 14.0% ± 6.1 | 28.8% ± 9.8 | 48.1% ± 12.1 | 59.3% ± 11.5 |
| Middle Eastern Woman | 2.8% ± 3.5 | 2.0% ± 2.6 | 1.2% ± 1.7 | 0.8% ± 1.3 | 0.4% ± 0.9 | 0.1% ± 0.4 | 0.0% ± 0.0 | 0.2% ± 0.4 | 0.5% ± 0.9 | 1.5% ± 2.1 | 2.8% ± 3.6 |
| White Man | 7.1% ± 6.4 | 5.2% ± 4.7 | 3.2% ± 2.9 | 2.1% ± 2.0 | 1.2% ± 1.2 | 0.3% ± 0.4 | 0.0% ± 0.0 | 0.4% ± 0.4 | 1.4% ± 1.3 | 4.2% ± 3.5 | 7.4% ± 6.4 |
| White Woman | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 |

## Files

- `beard_region_change_metrics.csv` — tidy raw metrics for plotting or statistical follow-up.
- `<category>/average_reference.png` — category-average baseline with representative ROI loop.
- `<category>/alpha_change_occurrence_jet.png` — requested blue→green→yellow→red frequency overlays.
- `<category>/alpha_blurred_mean_pixel_change.png` — blurred mean pixel-change/attenuation overlays.

The score is evidence of intervention-localized change, not a statement that every changed pixel is facial hair. A later semantic beard model can replace the thresholded-change signal while retaining the same landmark alignment, reporting, and visual layout.
