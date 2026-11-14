# SAE agreement with extracted concepts

Existing pretrained SAE; no training or suppression. Targets are unreviewed Florence-2 detector phrases, normalized by explicit patterns. Detector absence may be a missed detection.

72 images: 36 train, 36 validation; test untouched.

| Concept | Feature | Train AUC | Validation AUC | Matched groups train/val | Status |
|---|---:|---:|---:|---:|---|
| head_covering | 656 | 0.611 | 0.500 | 5/3 | within-group controls available; inspect validation effect |
| turban | 656 | 0.833 | 0.500 | 2/1 | within-group controls available; inspect validation effect |
| headscarf | 3364 | 0.667 | 0.721 | 2/2 | within-group controls available; inspect validation effect |
| earrings | 4013 | 0.556 | 0.500 | 2/4 | within-group controls available; inspect validation effect |
| necklace | 43 | 0.914 | 0.680 | 0/2 | insufficient within-group validation controls |
| beard | 3596 | 0.550 | 0.483 | 1/2 | within-group controls available; inspect validation effect |
| afro | 4015 | 1.000 | 0.833 | 1/0 | insufficient within-group validation controls |
| kimono | 1899 | 1.000 | 1.000 | 0/0 | insufficient within-group validation controls |
| shirt | 1949 | 0.696 | 0.627 | 7/3 | within-group controls available; inspect validation effect |
| tank_top | 4199 | 0.808 | 0.303 | 3/2 | within-group controls available; inspect validation effect |
| smile | 4911 | 0.667 | 0.450 | 3/4 | within-group controls available; inspect validation effect |
| flower | 1899 | 1.000 | 1.000 | 0/0 | insufficient within-group validation controls |
| red_dot | 576 | 0.667 | 0.500 | 2/1 | within-group controls available; inspect validation effect |

AUC 0.5 means chance ordering, 1 means perfect separation of detector positives/negatives. Validation labels never choose the feature. Group controls can contain only one positive/negative image in this small pilot; these are exploratory results with no significance claim. High global AUC can reflect prompt-group identity rather than the named concept. Missing matched validation groups prevents distinguishing those explanations. Features can mix concepts. No concept has been causally established.

Next: inspect spatial SAE maps, review/correct detector labels, expand within-group positive/negative controls, then test interventions and random-feature controls. Clothing and hairstyles are not themselves harm targets.

## Spatial replay check (completed pilot)

Twelve discovery examples were replayed with exact pixel identity to their saved images. Headscarf feature 3364 was inactive on both selected detector-positive images, so the AUC result does not support a reliable headscarf feature. Necklace feature 43 activated in a small region near the top of the head in the inspected example, rather than on the necklace. Kimono feature 1899 activated around the clothing/shoulder region in the inspected example, but kimono and flower detector labels are confounded with the same prompt group.

[Map measurements](maps/metrics.csv) and [map interpretation](maps/README.md). These observations weaken the first two candidates; the kimono association needs controls before it can be called concept-specific. No suppression has run.
