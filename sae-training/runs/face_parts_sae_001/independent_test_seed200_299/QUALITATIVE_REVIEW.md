# Qualitative review: ears, red dots, and SAE features

This is a first visual pass over the saved 100-image-per-prompt-group contact sheets, selected annotation overlays, and held-out SAE feature sheets. It is a qualitative review, not a complete per-image annotation audit.

## Findings

- **Ears:** visibility is often confounded by hair, head coverings, and pose. The contact sheets include examples where ears are covered by hair or scarves, and examples with turned heads. The ear heatmap for feature 3725 sometimes aligns with visible ears, but also lights up other face-side areas. Feature 4858 is broader still, often activating over cheeks and temples. Neither sheet supports calling either feature an ear detector.
- **Missing ear boxes:** phrase-grounding annotations are highly uneven across the 100 held-out images per prompt group: exact ear phrases occur in 0–25 images per group. The images and overlays show plausible reasons for some misses (covering, hair, pose), but this review does not establish that those explain all group differences. Missing annotations cannot be treated as absent ears.
- **Red dots:** the Indian-woman contact sheet visibly contains forehead marks in many images, while exact red-dot phrases occur in 51/100 annotations. For example, `indian_woman_seed200` visibly has a small forehead mark; its detector overlay contains two nested red-dot boxes, one much larger than the mark. Thus these annotations can be missed or imprecise, and should not be used as ground truth without review. Other prompt groups have far fewer red-dot phrases, so prompt construction is a strong confound.
- **Feature 3593:** in the held-out eye sheet, activation is generally positioned over the eyes across prompt groups. On the Indian-woman example (`seed228`), it is at the eye region rather than the forehead mark. The visual evidence does not suggest that 3593 is a red-dot feature. The sheet uses one selected seed per group and per-image normalized colors, so it cannot establish group-level activation differences.

## Recommended next analysis

Before estimating group differences, label a balanced subset of images for (1) ear visible / partly occluded / not visible, (2) approximate head pose, (3) earrings present / absent / unclear, and (4) red forehead mark present / absent / unclear. Review detector boxes against the image and retain uncertain labels. Then compare SAE activation on matched visible examples, including feature 3593 for the red-mark question and ear candidates 3725/4858 for ears. Keep raw activation magnitudes; do not compare heatmap colors normalized separately per image.

## Scope and files reviewed

- Group contact sheets in `contact_sheets/` (seed range 200–299).
- Annotation overlays for `indian_woman_seed200` and `white_woman_seed200`.
- `region_heatmaps/eyes/feature_3593.jpg`.
- `final_feature_localization/ears/feature_3725.jpg` and `feature_4858.jpg`.
- `annotations/bounding_boxes.csv` and `region_heatmaps/regional_activation_summary.csv`.

The bounding boxes and detector phrases are automatic and unreviewed. No model or data files were changed.
