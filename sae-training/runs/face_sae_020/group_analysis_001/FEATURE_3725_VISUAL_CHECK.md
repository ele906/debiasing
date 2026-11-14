# Feature 3725 visual check on additional images

Generated 10 new seeds (300–309) for each of the same 12 SDXL Turbo prompt groups,
using the original one-step recipe. The contact sheet shows the six women’s groups
(60 portraits) with feature 3725 activation overlays. All panels use one shared Jet
activation scale so colors are comparable across groups.

## Observation

Feature 3725 remains strongly activated on the Indian-woman prompt group. In the
additional images, its response often covers a broad upper-face area and can extend
over head coverings or adjacent clothing. It is not confined to a small forehead
mark. Middle-Eastern-woman images with head coverings also show noticeable activation,
though weaker in this small sample. The full-image mean activations were 16.91 for
Indian-woman prompts, 7.56 for Middle-Eastern-woman prompts, 1.17 for East-Asian-woman
prompts, 1.12 for Hispanic-woman prompts, and 0 for Black- and White-woman prompts.
These means use the full spatial map, not a face-only crop.

This visual check supports a repeatable prompt-group association. It does not identify
the feature as a bindi detector. Head coverings, clothing, forehead decorations, face
appearance, and other prompt-correlated details remain plausible explanations. The
next bindi and earring checks should use visually verified images and matched conditions.

## Artifacts

- `feature_maps/feature_3725_women_seed300_309_jet.jpg`: Jet overlay contact sheet.
- `feature_maps/feature_3725_women_seed300_309_summary.json`: mean activations and settings.
- 120 total new portraits were generated, 10 per prompt group, with residuals saved locally
  under `sae-training/runs/face_parts_sae_001/independent_test_seed300_309`.

The generation and rendering scripts are `sae-training/generate_face_sae_visual_check.sbatch`,
`sae-training/render_face_sae_visual_check.py`, and
`sae-training/render_face_sae_visual_check.sbatch`.
