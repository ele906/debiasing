# Feature-strength tuning

Start with the provisional necklace-region feature 3181. This pilot changes its contribution during SDXL Turbo generation, with the pretrained SAE and diffusion weights frozen. It also sweeps one seeded, activity-matched random feature control, excluding all earlier concept candidates.

The intervention is `h' = h + (alpha - 1) z_j d_j`, retaining the SAE reconstruction residual. Strengths 0, 0.5, 1, and 1.5 remove, attenuate, preserve, or amplify the feature contribution. Alpha 1 and saved baseline replays must be pixel-identical.

The first run uses all sixteen fresh necklace-control images: two prompt groups, four paired seeds, requested present/absent conditions. Requested conditions are not ground-truth labels. This is an exploratory intervention, not an established racism mitigation.

```bash
sbatch concept/tuning/run.slurm
```

`run_001/REPORT.md` contains target/control comparison grids; `metrics.csv` records image changes inside original necklace boxes, outside them, and inside face boxes. These pixel measurements do not determine whether a necklace disappeared. `review.csv` has blank fields for concept change, face/clothing damage, artifacts, and prompt fidelity. The automatic detector audit covers all distinct generated outputs, including baselines, and keeps human review fields blank.

Use a new `--output` when repeating sweeps. Do not pick a strength solely because a detector label disappears; inspect the corresponding image and matched control. No best strength is selected automatically.

## First completed run

[Results summary](run_001/SUMMARY.md) and [all target/control grids](run_001/REPORT.md).

Feature 3181 and activity-matched control 2808 were swept on sixteen fresh images. All sixteen saved-baseline replays and thirty-two alpha-1 hook outputs matched exactly. The detector audit covered 112 distinct image IDs. The target's necklace-label count stayed 9/16 at strengths 0, 0.5, 1, and 1.5; no baseline-positive labels were lost. Full control suppression changed the count to 10/16, with one gained label.

Visual spot checks on the Indian-man and Hispanic-woman requested-present examples at seed 1000 show necklace appearance changes alongside face/pose changes; necklaces remain visible. This demonstrates an influence on generation, rather than clean concept removal. Pixel changes and detector counts alone do not establish a good tuning strength. The complete human review sheet remains blank.

The automatic summary can be regenerated with:

```bash
python concept/tuning/summarize.py --run concept/tuning/run_001
```

`prefetch_weights.py` reads cached weights sequentially to reduce network-filesystem loading delays. It does not modify model files. All image generation and detector inference use cached models.
