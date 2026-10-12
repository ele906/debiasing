# SAE agreement with extracted concepts

Existing pretrained SAE; no training or suppression. Targets are unreviewed Florence-2 detector phrases, normalized by explicit patterns. Detector absence may be a missed detection.

960 images: 720 train, 240 validation; test rows are excluded from this comparison.

| Concept | Feature | Train AUC | Validation AUC | Within-group validation AUC | Matched groups train/val | Status |
|---|---:|---:|---:|---:|---:|---|
| head_covering | 4154 | 0.481 | 0.449 | 0.627 | 7/3 | within-group controls available; inspect validation effect |
| turban | 1526 | 0.618 | 0.619 | 0.357 | 2/1 | within-group controls available; inspect validation effect |
| headscarf | 1257 | 0.315 | 0.228 | 0.405 | 2/2 | within-group controls available; inspect validation effect |
| earrings | 4858 | 0.460 | 0.407 | 0.495 | 5/3 | within-group controls available; inspect validation effect |
| necklace | 3985 | 0.802 | 0.805 | 0.742 | 5/1 | within-group controls available; inspect validation effect |
| beard | 5085 | 0.909 | 0.910 | 0.422 | 4/2 | within-group controls available; inspect validation effect |
| afro | 1500 | 0.958 | 0.966 | 0.667 | 1/1 | within-group controls available; inspect validation effect |
| kimono | 5046 | 0.640 | 0.628 |  | 2/0 | insufficient within-group validation controls |
| shirt | 2858 | 0.651 | 0.615 | 0.527 | 8/7 | within-group controls available; inspect validation effect |
| tank_top | 318 | 0.597 | 0.451 | 0.500 | 3/2 | within-group controls available; inspect validation effect |
| smile | 3593 | 0.654 | 0.617 | 0.596 | 6/4 | within-group controls available; inspect validation effect |
| flower | 2230 | 0.859 | 0.859 | 0.548 | 2/1 | within-group controls available; inspect validation effect |
| red_dot | 1817 | 0.438 | 0.460 | 0.636 | 2/1 | within-group controls available; inspect validation effect |

AUC 0.5 means chance ordering, 1 means perfect separation of detector positives/negatives. Validation labels never choose the feature. Feature selection requires at least 3 detector positives and 3 negatives per matched training group, with group AUC weighted by the smaller class size. Features must activate on at least 5 detector-positive training images. These are exploratory results with no significance claim. High global AUC can reflect prompt-group identity rather than the named concept. Missing matched validation groups prevents distinguishing those explanations. Features can mix concepts. No concept has been causally established.

Next: inspect spatial SAE maps, review/correct detector labels, expand within-group positive/negative controls, then test interventions and random-feature controls. Clothing and hairstyles are not themselves harm targets.