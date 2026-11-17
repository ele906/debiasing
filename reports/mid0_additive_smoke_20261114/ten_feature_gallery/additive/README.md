# Additive interventions: men and women

Each sheet has six race-prompt rows and four columns: baseline, add+10, add+20, add+50. All images are saved seed30 generations, matching the earlier gallery. No diffusion generation or new model inference was performed. Original512×512 PNGs remain untouched; these are256px viewing derivatives.

Jet overlays show the magnitude of the selected **modified SAE coordinate**, not beard segmentation or causal attribution. Maps are16×16, enlarged using nearest-neighbor interpolation before thumbnailing. Zero activation is transparent; positive activation uses45% overlay opacity. Each feature has one shared scale across both genders and all four columns, with no per-image normalization or percentile clipping. Scales differ between features: read each colorbar before comparing colors.

Adding β at every spatial position can make the overlays broad even where natural activation was sparse. The baseline map comes from the paired pre-edit activation archive; its equality across the three offsets was verified. The heatmaps show latents before casting the decoded intervention to the model output dtype.

| Feature | Men | Women | Men + jet | Women + jet |
|---|---|---|---|---|
| 1074 | [Photos](figures/feature_1074_men.png) | [Photos](figures/feature_1074_women.png) | [Overlay](figures/feature_1074_men_jet.png) | [Overlay](figures/feature_1074_women_jet.png) |
| 1650 | [Photos](figures/feature_1650_men.png) | [Photos](figures/feature_1650_women.png) | [Overlay](figures/feature_1650_men_jet.png) | [Overlay](figures/feature_1650_women_jet.png) |
| 2023 | [Photos](figures/feature_2023_men.png) | [Photos](figures/feature_2023_women.png) | [Overlay](figures/feature_2023_men_jet.png) | [Overlay](figures/feature_2023_women_jet.png) |
| 2361 | [Photos](figures/feature_2361_men.png) | [Photos](figures/feature_2361_women.png) | [Overlay](figures/feature_2361_men_jet.png) | [Overlay](figures/feature_2361_women_jet.png) |
| 4228 | [Photos](figures/feature_4228_men.png) | [Photos](figures/feature_4228_women.png) | [Overlay](figures/feature_4228_men_jet.png) | [Overlay](figures/feature_4228_women_jet.png) |
| 4338 | [Photos](figures/feature_4338_men.png) | [Photos](figures/feature_4338_women.png) | [Overlay](figures/feature_4338_men_jet.png) | [Overlay](figures/feature_4338_women_jet.png) |
| 4399 | [Photos](figures/feature_4399_men.png) | [Photos](figures/feature_4399_women.png) | [Overlay](figures/feature_4399_men_jet.png) | [Overlay](figures/feature_4399_women_jet.png) |
| 4554 | [Photos](figures/feature_4554_men.png) | [Photos](figures/feature_4554_women.png) | [Overlay](figures/feature_4554_men_jet.png) | [Overlay](figures/feature_4554_women_jet.png) |
| 4586 | [Photos](figures/feature_4586_men.png) | [Photos](figures/feature_4586_women.png) | [Overlay](figures/feature_4586_men_jet.png) | [Overlay](figures/feature_4586_women_jet.png) |
| 5094 | [Photos](figures/feature_5094_men.png) | [Photos](figures/feature_5094_women.png) | [Overlay](figures/feature_5094_men_jet.png) | [Overlay](figures/feature_5094_women_jet.png) |

## Reproduce

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python config/code/additive_gallery.py
```

The script creates a new timestamped directory. config records input identity and code; results/panels.csv links each tile to its unchanged source image and activation archive, with hashes. maps/ retains the selected original and modified float32 feature maps losslessly. The original full activation archives remain at their recorded paths.

Full maps and original archives remain permanently at `/n/fs/goose/experiments/mid0_sae_selectivity/additive_gender_gallery_20261117_050815_555416`. This GitHub release contains the photo/overlay previews, code, CSV metadata and hashes; lossless activation NPZs remain on the research host.
