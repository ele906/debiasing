# C2C–SAE candidate intersections

All 240 validation images were replayed with exact pixel identity. No SAE or diffusion training or suppression. Feature IDs were fixed by the earlier train-only ranking. Detector labels and boxes remain unreviewed.

Spatial enrichment = fraction of feature activation inside matching boxes / fraction of image covered by those boxes. Uniform spatial activation has enrichment 1. Zero-activation maps are excluded from localization ratios and counted separately. Box coverage accounts for coarse SAE cells. Circular map translations provide an exploratory spatial reference, not a significance test.

| Concept | Feature | Within-group validation AUC | Detector positives | Active positives | Pooled spatial enrichment | Controlled spatial enrichment | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| necklace | 3985 | 0.742 | 52 | 48 | 4.240 | 0.813 | not supported by current checks |
| necklace | 4916 | 0.530 | 52 | 50 | 0.000 | 0.000 | not supported by current checks |
| necklace | 1187 | 0.571 | 52 | 45 | 3.542 | 0.000 | not supported by current checks |
| necklace | 3181 | 0.788 | 52 | 40 | 6.219 | 7.105 | provisional activation + controlled spatial intersection |
| necklace | 1257 | 0.465 | 52 | 52 | 0.000 | 0.000 | not supported by current checks |
| beard | 5085 | 0.422 | 60 | 56 | 3.537 | 3.580 | detector-negative controls need human review |
| beard | 1088 | 0.449 | 60 | 20 | 0.000 | 0.000 | detector-negative controls need human review |
| beard | 4051 | 0.414 | 60 | 60 | 0.000 | 0.000 | detector-negative controls need human review |
| beard | 452 | 0.484 | 60 | 60 | 2.876 | 3.506 | detector-negative controls need human review |
| beard | 2893 | 0.562 | 60 | 60 | 3.213 | 3.228 | detector-negative controls need human review |
| afro | 1500 | 0.667 | 17 | 17 | 1.574 | 1.574 | detector-negative controls need human review |
| afro | 4524 | 0.588 | 17 | 17 | 1.216 | 1.216 | detector-negative controls need human review |
| afro | 3579 | 0.471 | 17 | 17 | 1.199 | 1.199 | detector-negative controls need human review |
| afro | 3745 | 0.431 | 17 | 17 | 1.118 | 1.118 | detector-negative controls need human review |
| afro | 2092 | 0.412 | 17 | 17 | 0.986 | 0.986 | detector-negative controls need human review |

The status is a descriptive screen: within-group validation AUC above 0.5 and median spatial enrichment above 1 inside the same qualifying validation groups (at least three detector positives and three negatives). Pooled enrichment from other groups cannot validate that intersection. It is not semantic ground truth or causal evidence. A concept with one qualifying group has limited controls. Many features/concepts are inspected, and validation has been used iteratively; confirmatory conclusions need new independent examples.

## Matched visual examples

Each figure uses one race/gender prompt group containing both detector-positive and detector-negative validation images. Choose the group with the most balanced classes, then the first two examples per class in manifest order. This selection does not use SAE activation or overlap. Blue boxes are detector regions; the right panel combines boxes and the SAE map. Maps share one scale across the four examples for a concept. Negatives can be detector misses.

### necklace, feature 3985 — indian man

![necklace detector and SAE comparison](figures/necklace_feature3985.jpg)

### necklace, feature 4916 — indian man

![necklace detector and SAE comparison](figures/necklace_feature4916.jpg)

### necklace, feature 1187 — indian man

![necklace detector and SAE comparison](figures/necklace_feature1187.jpg)

### necklace, feature 3181 — indian man

![necklace detector and SAE comparison](figures/necklace_feature3181.jpg)

### necklace, feature 1257 — indian man

![necklace detector and SAE comparison](figures/necklace_feature1257.jpg)

### beard, feature 5085 — east asian man

![beard detector and SAE comparison](figures/beard_feature5085.jpg)

### beard, feature 1088 — east asian man

![beard detector and SAE comparison](figures/beard_feature1088.jpg)

### beard, feature 4051 — east asian man

![beard detector and SAE comparison](figures/beard_feature4051.jpg)

### beard, feature 452 — east asian man

![beard detector and SAE comparison](figures/beard_feature452.jpg)

### beard, feature 2893 — east asian man

![beard detector and SAE comparison](figures/beard_feature2893.jpg)

### afro, feature 1500 — black woman

![afro detector and SAE comparison](figures/afro_feature1500.jpg)

### afro, feature 4524 — black woman

![afro detector and SAE comparison](figures/afro_feature4524.jpg)

### afro, feature 3579 — black woman

![afro detector and SAE comparison](figures/afro_feature3579.jpg)

### afro, feature 3745 — black woman

![afro detector and SAE comparison](figures/afro_feature3745.jpg)

### afro, feature 2092 — black woman

![afro detector and SAE comparison](figures/afro_feature2092.jpg)

## Visual review findings

The top-one necklace feature 3985 activated around the mouth in the inspected matched Indian-man examples, rather than the necklace. The two inspected detector-negative afro examples (black_woman_seed042 and black_woman_seed051) visibly contain the hairstyle. The two inspected detector-negative beard examples (east_asian_man_seed002 and east_asian_man_seed012) visibly contain facial hair. These are spot checks, not a complete annotation pass; raw detector labels are preserved. Afro and beard controls cannot support a confirmed semantic distinction until corrected. numeric_screen_pass records the quantitative screen separately from these observed detector-quality issues.

## What remains

Review/correct boxes and labels, inspect whether features localize consistently across prompt groups, collect independent controls, then test attenuation with random-feature controls. Spatial overlap alone can result from nearby or correlated content and does not establish that a feature exclusively represents the concept.
