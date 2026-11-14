# Feature 1650 — landmark-constrained lower-face change report

For each alpha image and seed, MediaPipe Face Landmarker constructs a lower-face/extended-chin polygon. The jet overlay is the pixelwise mean of those polygons: its intensity is the fraction of seeds for which the landmarker selected that pixel as part of the beard-region geometry. This is not a semantic beard segmenter.

The cyan contour in each jet panel marks 50% selection frequency. The geometry extends below landmark 152 (chin) so long beards are not clipped. The `alpha_blurred_mean_pixel_change.png` panel is separate attenuation evidence: seed-level RGB-MAE maps are blurred before averaging. Its pixel-change threshold is **12.0** (0–255 scale).

## Mean landmark beard-region area by alpha

Each cell is mean landmark-selected beard-region area, normalized to the `+1` reference's mean area, followed by its across-seed standard deviation. Unlike the old change map, `+1` is expected to show a nonzero region here.

| Category | -7 | -5 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +5 | +7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Black Man | 100.1% ± 21.1 | 100.1% ± 21.1 | 100.1% ± 21.1 | 100.1% ± 21.2 | 100.1% ± 21.2 | 100.0% ± 21.2 | 100.0% ± 21.3 | 100.0% ± 21.3 | 100.0% ± 21.2 | 100.0% ± 21.2 | 100.0% ± 21.3 |
| Black Woman | 101.2% ± 33.5 | 100.9% ± 33.8 | 100.7% ± 33.8 | 100.5% ± 34.0 | 100.5% ± 33.8 | 100.3% ± 33.8 | 100.0% ± 33.9 | 99.7% ± 33.8 | 99.5% ± 33.7 | 99.4% ± 33.5 | 99.6% ± 33.6 |
| East Asian Man | 99.3% ± 27.4 | 99.5% ± 27.5 | 99.7% ± 27.5 | 99.7% ± 27.5 | 99.8% ± 27.5 | 99.9% ± 27.6 | 100.0% ± 27.7 | 100.0% ± 27.7 | 100.1% ± 27.8 | 100.2% ± 27.8 | 100.3% ± 27.8 |
| East Asian Woman | 99.9% ± 29.3 | 99.9% ± 29.3 | 99.9% ± 29.2 | 99.9% ± 29.3 | 100.0% ± 29.3 | 100.0% ± 29.3 | 100.0% ± 29.3 | 100.0% ± 29.3 | 100.0% ± 29.3 | 100.1% ± 29.3 | 100.1% ± 29.3 |
| Hispanic Man | 100.0% ± 33.2 | 100.1% ± 33.2 | 100.1% ± 33.2 | 100.1% ± 33.2 | 100.1% ± 33.2 | 100.0% ± 33.2 | 100.0% ± 33.2 | 100.0% ± 33.2 | 100.0% ± 33.3 | 100.0% ± 33.3 | 100.0% ± 33.3 |
| Hispanic Woman | 100.2% ± 35.7 | 100.1% ± 35.7 | 100.1% ± 35.7 | 100.2% ± 35.7 | 100.1% ± 35.7 | 100.1% ± 35.7 | 100.0% ± 35.7 | 100.1% ± 35.7 | 100.0% ± 35.7 | 100.1% ± 35.7 | 100.0% ± 35.8 |
| Indian Man | 100.0% ± 17.6 | 100.0% ± 17.7 | 100.0% ± 17.7 | 99.9% ± 17.6 | 99.9% ± 17.6 | 100.0% ± 17.6 | 100.0% ± 17.7 | 100.1% ± 17.7 | 100.0% ± 17.6 | 100.0% ± 17.7 | 100.0% ± 17.6 |
| Indian Woman | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 | 100.0% ± 21.6 |
| Middle Eastern Man | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 | 100.0% ± 17.9 |
| Middle Eastern Woman | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 | 100.0% ± 19.0 |
| White Man | 100.1% ± 16.5 | 100.1% ± 16.5 | 100.1% ± 16.4 | 100.1% ± 16.4 | 100.1% ± 16.4 | 100.1% ± 16.4 | 100.0% ± 16.3 | 100.0% ± 16.4 | 99.9% ± 16.4 | 100.0% ± 16.4 | 100.0% ± 16.3 |
| White Woman | 99.7% ± 36.1 | 99.8% ± 35.9 | 99.7% ± 35.7 | 99.9% ± 35.8 | 99.9% ± 35.6 | 100.0% ± 35.6 | 100.0% ± 35.5 | 99.9% ± 35.5 | 100.0% ± 35.5 | 99.9% ± 35.3 | 100.0% ± 35.1 |

## Files

- `beard_region_change_metrics.csv` — tidy raw metrics for plotting or statistical follow-up.
- `<category>/average_reference.png` — category-average baseline with representative ROI loop.
- `<category>/alpha_landmark_beard_region_frequency_jet.png` — blue→green→yellow→red landmarker-selection-frequency overlays; no in-grid legend cell.
- `<category>/alpha_blurred_mean_pixel_change.png` — blurred mean pixel-change/attenuation overlays.

The jet score is facial geometry selected by landmarks, not a statement that every selected pixel is facial hair. The separate attenuation panel is evidence of intervention-localized RGB change. A later semantic beard model can replace the geometric region while retaining the same report layout.
