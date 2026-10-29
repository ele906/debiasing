# Expanded visual review and spatial check

## Visual labels

All 80 generated pilot images were reviewed from the shuffled contact sheets. Labels: **45 clear bindi-visible, 12 no clear red bindi, 23 uncertain**. Ambiguous small/dark/faint marks were kept as uncertain. Several images prompted with no bindi still show a red bindi-like mark, and some prompted bindi images have multiple forehead ornaments; prompt condition is not treated as visual truth.

These are generated images and visual annotations from contact sheets, not human-subject data. The review is visual but not a formal independent blinded study.

## Feature 3725 spatial check

The saved residual grids were re-encoded with the saved face SAE to recover 16×16 activation maps. In the original eight-image check, feature 3725 was stronger for visible-bindi images in the broad forehead average. However, its spatial maps are not confined to the bindi: strong activation also appears on cheeks, hair/head edges, and other face areas. The map maxima are usually away from the bindi. The center-forehead patch is not the dominant activation area. This weakens the interpretation that feature 3725 is a bindi detector. See `../feature_3725_spatial_overlay.png`.

## Expanded activation comparison

Across the reviewed clear-label images, feature 3725’s per-seed mean forehead activation difference (visible bindi minus no clear red bindi) is +8.52 over 10 seeds with both label types (seed-bootstrap 95% interval +7.44 to +9.66). Feature 3593’s corresponding difference is +4.92 over 10 seed pairs (95% interval +4.04 to +5.79). These intervals do not remove prompt/style confounding and are descriptive only.

Prompt adherence is a major confound: all 40 bindi-prompt images were visibly bindi-positive; among the 40 no-bindi-prompt images, 5 were visibly positive, 12 had no clear red bindi, and 23 were uncertain. Within only the no-bindi-prompt images, the five positive examples have mean feature 3725 activation 3.56 versus 0.82 for the 12 clear negatives. Those five positives come from only three seeds, and only one seed has both visible labels (with the earring prompt changing), so this is a weak, unpaired clue.

Bindi presence also co-occurs with sindoor, head jewelry, earrings, clothing, lighting, and other changes. The same seed does not hold the rendered image fixed. Therefore this analysis does **not** identify a bindi-specific feature or justify feature ablation. It does not test race or skin-color effects.

## Attempt at a cleaner pair

A localized image-generation edit was tried on a clear no-bindi portrait. It produced a noticeably different, closer crop and regenerated facial details. It was excluded from activation analysis because it does not hold the portrait fixed. The next useful experiment needs a same-image local edit that preserves the surrounding portrait, followed by visual verification before computing activations. The feature intervention should wait until that comparison is available.

## Artifacts

- `visual_labels.csv`: image-level labels and prompt conditions.
- `seed_paired_differences.csv`: per-seed forehead activation differences for features 3725 and 3593.
- `feature_group_summary.csv`: largest absolute group differences among all 5,120 features (screening only).
- `prompt_adherence.csv`: visual bindi labels by prompt setting.
- `top_feature_activation_heatmap.png`: JET heatmap, SAE features on the y-axis.
- `summary.json`: compact counts and statistics.
