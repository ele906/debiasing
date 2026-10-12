# Quick research update

We are identifying candidate concepts shared by a C2C-style Florence-2 audit and the pretrained SDXL Turbo SAE. The current evidence is exploratory. A first feature-strength intervention has now run; it did not remove necklace detector labels.

## Completed

- Extracted concepts from all 1,200 original images across 12 prompt categories.
- Generated complete Markdown count tables and bounding-box reports.
- Replayed all 240 validation images exactly and compared 13 candidate SAE features with concept boxes.
- Checked all five previously train-selected candidates for necklace, afro, and beard: 15 feature–concept pairs. All results, including failures, are retained.

## Main findings

| Concept | Feature | Finding |
|---|---:|---|
| Necklace region | 3181 | Provisional intersection: within-group validation AUC 0.788; median spatial concentration 7.11 times the box-area baseline in the qualifying group. Also activates on negatives; not an exclusive necklace feature. |
| Necklace | 3985 | Earlier candidate failed the controlled spatial check; inspected maps activated near the mouth. |
| Afro | 1500 | Spatial overlap exists, but inspected detector-negative controls visibly contain rounded curly hair. Specificity needs reviewed controls. |
| Beard | 2893 | Localizes around facial hair; within-group AUC only 0.563 and detector-negative controls contain facial hair. Not confirmed as concept-specific. |

The necklace result comes from one qualifying validation prompt group and eleven detector-positive images in that group. The spatial ratio is relative to the area of the boxes, not a positive/negative activation ratio. Many candidates have been inspected; new independent controls are needed for stronger conclusions.

## Fresh controls

Generated 48 new paired images using four new seeds per group, two groups per concept, and requested present/absent conditions for necklace, afro/straight hair, and beard/clean-shaven faces. Requested conditions are not ground-truth labels. These images are an additional control set and have not yet been treated as validated evidence.

## For the presentation

> I compared concepts detected in images with SAE feature activations and spatial maps. I found a provisional necklace-region overlap, while the checks also exposed confounded features and detector misses. The next step is to review fresh controls and test whether the overlap reflects the concept rather than nearby visual content.

## Evidence

- [All original candidate spatial checks](intersections/REPORT.md)
- [All fifteen alternative-candidate checks](intersections_alternatives/REPORT.md)
- [Necklace feature 3181: matched images and boxes](intersections_alternatives/figures/necklace_feature3181.jpg)
- [Recorded visual review issues](intersections/visual_review.md)
- [Fresh necklace controls](independent_controls/necklace_controls.jpg)
- [Fresh afro controls](independent_controls/afro_controls.jpg)
- [Fresh beard controls](independent_controls/beard_controls.jpg)

## First tuning experiment

[Completed tuning summary](tuning/run_001/SUMMARY.md). Feature 3181 and an activity-matched random control were swept at alpha 0, 0.5, 1, and 1.5 on sixteen fresh images. Every unchanged-strength output matched the original exactly. Necklace detector presence stayed 9/16 across target strengths. Inspected examples show necklace appearance and face/pose changes, not clean necklace removal. This is a useful distinction between finding a spatial association and achieving concept-specific control.
