# Region-specific epoch heatmaps

Each sheet shows one region-specific SAE feature on held-out test images. Rows
are the 12 race/gender prompt categories; columns are checkpoints at epochs 0,
4, 8, 12, 16, and 20. The pixelated `jet` activation overlay is clipped to
the matching automatically annotated region box, outlined in cyan.

| Region | Validation-selected features | Categories with a usable test box |
|---|---|---:|
| Eyes | 3701, 3593, 1600 | 9 / 12 |
| Eyebrows | 1600, 3701, 1237 | 7 / 12 |
| Nose | 1664, 3593, 1991 | 4 / 12 |
| Ears | 4858, 2230, 2611 | 6 / 12 |

Each feature ID has its own JPEG in the matching region folder. Candidate
features were selected using validation activation contrast, then visualized
on test images. Missing rows mean there was no accepted detector box and SAE
residual for that test category. Boxes are automatic and unreviewed; the
overlays show spatial activation association, not proof of semantic meaning.

`settings.json` records feature IDs, checkpoints, box filters, and selected
test sample IDs. The generator is not part of these overlays; they show SAE
activations on the original held-out images.
