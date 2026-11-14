# Global facial-feature seed-average analysis

This report contains the compact visual summaries for global mid.0 interventions on features 2023, 1650, and 4228. It uses the face-focused SAE, the biased/simple-prompt dataset, all 12 categories, and fresh seeds 30–59.

Each feature folder includes, for every category:

- `*_mean_blur_grid.png`: blurred RGB mean across 30 seeds, ordered original then alpha −7 through +7.
- `*_mean_pixel_difference_grid.png`: signed mean luminance change versus the original/control mean; blue is darker and red is brighter.
- `*_mean_sae_overlay_grid.png`: mean SAE coefficient-change overlay on the blurred mean; blue indicates suppression and red boosting.
- `all_categories_*_matrix.png`: 12 categories as rows and original/alpha values as columns.

Raw generated images and per-seed sheets are intentionally not included because they are multi-gigabyte; the source run retains them locally.
