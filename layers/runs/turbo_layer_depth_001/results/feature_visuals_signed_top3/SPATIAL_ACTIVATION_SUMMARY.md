# Spatial activation-area summary

This summarizes how much of the SAE spatial grid is active for the displayed
Top-3 feature maps. It explains why the down.2.1 contact-sheet overlays look
larger and more global than the mid/up overlays.

## Measurement

- **Samples:** all 12 prompt groups, held-out seed 150.
- **Features:** ranks 1–3, selected on discovery seeds 100–149 only.
- **Grid:** every measured map is 16 × 16 SAE cells.
- **Area:** fraction of cells with positive baseline SAE activation (`z_f > 0`),
  averaged equally over the 36 displayed layer/group/rank maps per layer.

This is a spatial-coverage measurement, not image-pixel area, activation
magnitude, semantic importance, fairness, or image quality.

| Layer | Mean active map area |
|---|---:|
| down.2.1 | 54.4% |
| mid.0 | 7.0% |
| up.0.0 | 9.5% |
| up.0.1 | 15.0% |

## By discovery rank

| Layer | Rank 1 mean | Rank 2 mean | Rank 3 mean |
|---|---:|---:|---:|
| down.2.1 | 81.0% | 50.7% | 31.5% |
| mid.0 | 8.5% | 6.7% | 5.7% |
| up.0.0 | 11.9% | 8.6% | 8.1% |
| up.0.1 | 14.7% | 17.8% | 12.4% |

## Interpretation

On this held-out display seed, selected down.2.1 features are substantially
more spatially dense: rank-1 down.2.1 maps occupy about 81% of the 16 × 16
grid on average. Selected mid and up features are considerably more localized.
This is consistent with the contact sheets, but it is a descriptive result for
these selected features and this seed—not a general claim about every feature
in each layer.

The JET color scale is feature-wise normalized, so use this table—not apparent
color brightness—to compare how much map area is active across layers.
