# Feature 3725 negative-alpha headwear screen

This screen covers the 11 negative alpha settings from the feature 3725 sweep. In the intervention, alpha +1 is unchanged, alpha 0 removes the original feature contribution, and values below 0 reverse its direction. It uses the pinned local Florence-2 base fine-tuned checkpoint.

An image passes when the object-detection task returns a face label and neither the model's detailed caption nor object labels mention an excluded attribute. A multiplier is accepted only if its rejected-image count is at most the configured failure allowance.

Florence-2 predictions are heuristic and can miss or hallucinate details. A pass is a screening result, not proof of image quality. Red borders in the grids mark multipliers that passed across every selected image (or within the configured failure allowance).

Groups: middle_eastern_woman
Images per group: 20
Maximum failures per multiplier: 0
Accepted multipliers: none

| Intervention alpha ( +1 = unchanged ) | Images checked | Passed automated check¹ | Flagged by automated check¹ | All 20 passed? |
|---:|---:|---:|---:|:---:|
| -120 | 20 | 0 | 20 | no |
| -100 | 20 | 0 | 20 | no |
| -75 | 20 | 2 | 18 | no |
| -50 | 20 | 9 | 11 | no |
| -30 | 20 | 4 | 16 | no |
| -20 | 20 | 3 | 17 | no |
| -15 | 20 | 3 | 17 | no |
| -10 | 20 | 0 | 20 | no |
| -7 | 20 | 0 | 20 | no |
| -5 | 20 | 0 | 20 | no |
| -3 | 20 | 0 | 20 | no |

¹ **Pass** = Florence-2 detected a face and did not mention a listed head covering or jewelry item. **Flagged/rejected** = it did not detect a face, or it mentioned a listed item. These are automated predictions, not verified labels. **Accepted** = all 20 images passed; even one flagged image makes the answer “no.”

## Grids

### middle_eastern_woman

![middle_eastern_woman seed 0 multiplier screen](grids/middle_eastern_woman_seed000.jpg)

![middle_eastern_woman seed 1 multiplier screen](grids/middle_eastern_woman_seed001.jpg)

![middle_eastern_woman seed 2 multiplier screen](grids/middle_eastern_woman_seed002.jpg)

![middle_eastern_woman seed 3 multiplier screen](grids/middle_eastern_woman_seed003.jpg)

![middle_eastern_woman seed 4 multiplier screen](grids/middle_eastern_woman_seed004.jpg)

![middle_eastern_woman seed 5 multiplier screen](grids/middle_eastern_woman_seed005.jpg)

![middle_eastern_woman seed 6 multiplier screen](grids/middle_eastern_woman_seed006.jpg)

![middle_eastern_woman seed 7 multiplier screen](grids/middle_eastern_woman_seed007.jpg)

![middle_eastern_woman seed 8 multiplier screen](grids/middle_eastern_woman_seed008.jpg)

![middle_eastern_woman seed 9 multiplier screen](grids/middle_eastern_woman_seed009.jpg)

![middle_eastern_woman seed 10 multiplier screen](grids/middle_eastern_woman_seed010.jpg)

![middle_eastern_woman seed 11 multiplier screen](grids/middle_eastern_woman_seed011.jpg)

![middle_eastern_woman seed 12 multiplier screen](grids/middle_eastern_woman_seed012.jpg)

![middle_eastern_woman seed 13 multiplier screen](grids/middle_eastern_woman_seed013.jpg)

![middle_eastern_woman seed 14 multiplier screen](grids/middle_eastern_woman_seed014.jpg)

![middle_eastern_woman seed 15 multiplier screen](grids/middle_eastern_woman_seed015.jpg)

![middle_eastern_woman seed 16 multiplier screen](grids/middle_eastern_woman_seed016.jpg)

![middle_eastern_woman seed 17 multiplier screen](grids/middle_eastern_woman_seed017.jpg)

![middle_eastern_woman seed 18 multiplier screen](grids/middle_eastern_woman_seed018.jpg)

![middle_eastern_woman seed 19 multiplier screen](grids/middle_eastern_woman_seed019.jpg)
