# Feature 4228 — upper/middle/lower face-box intervention

This test asks whether feature 4228's broad face effect becomes more specific when the same `+7` boost is applied only in one vertical third of the automatic `human face` box.

- **Dataset:** biased/simple-prompt images, held-out seeds 20–29, 12 prompt categories.
- **SAE:** face-focused mid.0 SAE; the diffusion model is unchanged.
- **Intervention:** feature 4228 at `+1` control and `+7` boost, spatially limited to the upper, middle, or lower third of the automatic face box.
- **Important limitation:** these automatic face boxes are broad and may include hair, beard, or headwear; “upper/middle/lower” is a coarse spatial experiment, not facial-part segmentation.

## Readable comparison pages

Each row uses one held-out seed, selected because it had the largest full-feature change in that prompt group.  Columns show the same source seed: **original | upper-third boost | middle-third boost | lower-third boost**.

- [All men comparison](all_men_upper_middle_lower_plus7.png)
- [All women comparison](all_women_upper_middle_lower_plus7.png)

## Raw data

- [Upper-third settings and metrics](upper/)
- [Middle-third settings and metrics](middle/)
- [Lower-third settings and metrics](lower/)

The purpose is to decide whether a feature bundle should target a particular subregion.  If one region produces a clearer effect with less collateral change, bundle candidates should be selected from features active in that same region—not from all face-localized features indiscriminately.
