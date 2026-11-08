# Feature 2023 — unconstrained full-image change report

For each seed and alpha, a pixel is counted when its RGB MAE from that seed's baseline is at least **12.0** (0–255 scale). The jet maps show the fraction of seeds in which each full-image pixel crossed that threshold.

This can establish spatially consistent visual change, including changes outside the face. It does **not** identify beard, hair, hat, hijab, or any other semantic object; those claims require a semantic detector.

## Mean changed full-image area by alpha

Each cell is the mean changed full-image area followed by across-seed SD. `+1` is the reference reconstruction.

| Category | -7 | -5 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +5 | +7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Black Man | 0.4% ± 0.9 | 0.2% ± 0.6 | 0.1% ± 0.4 | 0.1% ± 0.3 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.1 | 0.1% ± 0.3 | 0.2% ± 0.6 |
| Black Woman | 0.9% ± 1.6 | 0.6% ± 1.2 | 0.4% ± 0.8 | 0.3% ± 0.6 | 0.2% ± 0.4 | 0.1% ± 0.2 | 0.0% ± 0.0 | 0.1% ± 0.2 | 0.2% ± 0.4 | 0.4% ± 0.7 | 0.7% ± 1.2 |
| East Asian Man | 6.1% ± 7.9 | 4.5% ± 5.9 | 3.0% ± 4.1 | 2.2% ± 3.1 | 1.3% ± 1.9 | 0.5% ± 0.8 | 0.0% ± 0.0 | 0.6% ± 1.0 | 1.7% ± 2.6 | 4.5% ± 6.8 | 6.9% ± 10.1 |
| East Asian Woman | 6.7% ± 5.9 | 5.3% ± 4.8 | 3.7% ± 3.5 | 2.8% ± 2.7 | 1.8% ± 1.8 | 0.8% ± 0.9 | 0.0% ± 0.0 | 0.8% ± 0.9 | 1.9% ± 1.9 | 4.4% ± 4.5 | 6.8% ± 6.7 |
| Hispanic Man | 9.0% ± 12.9 | 7.5% ± 12.1 | 5.8% ± 10.6 | 4.7% ± 9.4 | 3.2% ± 7.5 | 1.2% ± 3.1 | 0.0% ± 0.0 | 0.9% ± 1.7 | 2.4% ± 3.5 | 5.3% ± 6.9 | 7.8% ± 9.5 |
| Hispanic Woman | 18.7% ± 17.8 | 16.2% ± 16.3 | 13.0% ± 14.3 | 10.6% ± 12.2 | 7.3% ± 8.8 | 3.3% ± 4.6 | 0.0% ± 0.0 | 2.7% ± 3.5 | 5.5% ± 6.0 | 9.9% ± 9.0 | 13.5% ± 11.2 |
| Indian Man | 20.7% ± 8.2 | 17.0% ± 7.4 | 12.5% ± 5.9 | 9.7% ± 4.9 | 6.4% ± 3.5 | 2.8% ± 1.9 | 0.0% ± 0.0 | 2.6% ± 1.7 | 6.1% ± 3.2 | 12.2% ± 5.2 | 17.3% ± 7.2 |
| Indian Woman | 16.6% ± 7.7 | 13.2% ± 6.6 | 9.2% ± 5.2 | 6.9% ± 4.2 | 4.4% ± 2.8 | 1.7% ± 1.2 | 0.0% ± 0.0 | 1.7% ± 1.1 | 4.2% ± 2.5 | 8.9% ± 4.6 | 13.0% ± 5.8 |
| Middle Eastern Man | 33.0% ± 12.5 | 27.3% ± 11.4 | 21.1% ± 10.0 | 17.0% ± 8.6 | 11.9% ± 6.6 | 5.8% ± 3.6 | 0.0% ± 0.0 | 5.8% ± 3.5 | 11.4% ± 5.6 | 21.3% ± 8.6 | 28.6% ± 10.8 |
| Middle Eastern Woman | 25.3% ± 9.2 | 21.0% ± 8.6 | 16.1% ± 7.5 | 13.0% ± 6.5 | 9.2% ± 4.9 | 4.7% ± 2.8 | 0.0% ± 0.0 | 4.4% ± 2.3 | 8.7% ± 3.9 | 15.5% ± 6.2 | 20.9% ± 7.3 |
| White Man | 0.4% ± 1.7 | 0.3% ± 1.1 | 0.1% ± 0.6 | 0.1% ± 0.3 | 0.0% ± 0.2 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.1 | 0.1% ± 0.3 | 0.2% ± 0.6 |
| White Woman | 0.2% ± 0.6 | 0.1% ± 0.4 | 0.1% ± 0.1 | 0.0% ± 0.1 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.0% ± 0.0 | 0.1% ± 0.2 | 0.1% ± 0.4 |

## Files

- `full_image_change_metrics.csv` — tidy measurements.
- `<category>/alpha_full_image_change_frequency_jet.png` — seed-consistency overlay.
- `<category>/alpha_blurred_mean_pixel_change.png` — attenuation/change overlay.
