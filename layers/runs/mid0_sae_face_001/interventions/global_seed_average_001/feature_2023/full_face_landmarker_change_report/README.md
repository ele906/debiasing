# Feature 2023 — landmark-constrained full-face change report

MediaPipe Face Landmarker runs on every generated alpha image. Its complete face-oval mesh forms a soft full-face mask; no lower-face-only region is used. The cyan line in each grid is the 50% mask-frequency contour.

Jet intensity is the fraction of seeds whose pixel changes by RGB MAE ≥ **12.0** from its seed baseline, within that alpha image's full-face mask. This establishes spatially repeated visual change, not a semantic label such as forehead covering.

## Mean changed full-face area by alpha

| Category | -7 | -5 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +5 | +7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Black Man | 0.3% ± 0.9 | 0.2% ± 0.5 | 0.1% ± 0.3 | 0.1% ± 0.2 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.1 | 0.1% ± 0.2 | 0.1% ± 0.3 |
| Black Woman | 1.9% ± 2.7 | 1.3% ± 1.8 | 0.7% ± 1.0 | 0.5% ± 0.7 | 0.3% ± 0.5 | 0.1% ± 0.2 | 0.0% ± 0.0 | 0.1% ± 0.2 | 0.3% ± 0.5 | 0.7% ± 1.1 | 1.2% ± 1.9 |
| East Asian Man | 8.1% ± 9.6 | 6.0% ± 7.3 | 3.7% ± 5.0 | 2.5% ± 3.7 | 1.4% ± 2.2 | 0.4% ± 0.7 | 0.0% ± 0.0 | 0.5% ± 0.8 | 1.7% ± 2.7 | 4.9% ± 6.7 | 8.1% ± 10.0 |
| East Asian Woman | 8.1% ± 9.0 | 6.1% ± 6.9 | 4.0% ± 4.7 | 2.9% ± 3.6 | 1.8% ± 2.3 | 0.6% ± 1.0 | 0.0% ± 0.0 | 0.6% ± 0.9 | 1.9% ± 2.6 | 5.0% ± 5.8 | 8.3% ± 9.0 |
| Hispanic Man | 10.4% ± 13.0 | 8.6% ± 12.2 | 6.7% ± 10.6 | 5.4% ± 9.1 | 3.7% ± 6.9 | 1.2% ± 2.0 | 0.0% ± 0.0 | 1.2% ± 2.0 | 3.4% ± 5.0 | 7.8% ± 10.6 | 11.4% ± 14.3 |
| Hispanic Woman | 14.9% ± 13.4 | 12.4% ± 11.8 | 8.9% ± 9.2 | 6.9% ± 7.7 | 4.4% ± 5.3 | 1.5% ± 2.0 | 0.0% ± 0.0 | 1.4% ± 2.2 | 3.9% ± 5.9 | 9.3% ± 11.3 | 14.5% ± 15.7 |
| Indian Man | 18.1% ± 11.8 | 14.8% ± 10.9 | 10.9% ± 9.1 | 8.2% ± 7.5 | 5.2% ± 5.4 | 2.2% ± 2.7 | 0.0% ± 0.0 | 2.2% ± 2.9 | 5.3% ± 5.6 | 11.4% ± 8.9 | 16.6% ± 11.1 |
| Indian Woman | 15.5% ± 11.7 | 11.9% ± 9.7 | 8.0% ± 7.0 | 5.7% ± 5.3 | 3.4% ± 3.5 | 1.2% ± 1.4 | 0.0% ± 0.0 | 1.1% ± 1.1 | 3.1% ± 2.7 | 7.5% ± 5.9 | 12.0% ± 8.7 |
| Middle Eastern Man | 23.3% ± 12.3 | 18.2% ± 11.7 | 12.8% ± 10.4 | 9.6% ± 9.1 | 5.8% ± 6.6 | 2.1% ± 3.0 | 0.0% ± 0.0 | 1.9% ± 2.5 | 5.2% ± 5.4 | 12.3% ± 9.4 | 19.0% ± 11.8 |
| Middle Eastern Woman | 18.8% ± 9.4 | 15.1% ± 8.4 | 11.0% ± 7.1 | 8.5% ± 5.9 | 5.7% ± 4.3 | 2.5% ± 2.2 | 0.0% ± 0.0 | 2.4% ± 2.2 | 5.4% ± 4.0 | 10.3% ± 6.5 | 14.5% ± 8.0 |
| White Man | 0.7% ± 3.4 | 0.4% ± 2.3 | 0.2% ± 1.2 | 0.1% ± 0.7 | 0.1% ± 0.4 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.1 | 0.1% ± 0.3 | 0.1% ± 0.6 | 0.2% ± 0.8 |
| White Woman | 0.3% ± 0.7 | 0.2% ± 0.5 | 0.1% ± 0.3 | 0.1% ± 0.2 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.1 | 0.1% ± 0.3 | 0.2% ± 0.6 |

- `full_face_change_metrics.csv` — tidy data.
- `<category>/alpha_full_face_change_frequency_jet.png` — full-face change-frequency maps.
- `<category>/alpha_blurred_mean_pixel_change.png` — masked attenuation maps.
