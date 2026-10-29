# Feature intervention pilot readout

## Question

Do feature interventions visibly change a candidate concept while preserving a recognizable face, and does one multiplier work across images?

## Pilot setup

We screened saved SDXL-Turbo outputs for 20 Middle Eastern-woman prompts per feature. Feature 138 used alpha values -5, 0, +5, and +10. Feature 3725 used 0, +5, +10, +15, and +20. The intervention defines alpha +1 as the no-op; alpha 0 suppresses the selected feature coefficient. Florence-2 checked for a detected face and whether captions or object labels mentioned earrings or headwear. This is a heuristic screen, not ground truth.

## Results

| Feature | Initial concept hypothesis | Face detected by checker | Images with no headwear/earring terms at tested alphas | Spot check of seed 0 |
|---:|---|---:|---|---|
| 138 | Headwear or headwear styling | 20/20 at each alpha | -5: 3/20; 0: 0/20; +5: 0/20; +10: 0/20 | Face stays recognizable. The covering changes style, but remains present at all four settings. |
| 3725 | Headwear, forehead decoration, or jewelry | 20/20 at each alpha | 0: 0/20; +5: 4/20; +10: 9/20; +15: 12/20; +20: 13/20 | Face stays recognizable and the head covering remains. Jewelry/forehead adornment is still visible, including at +20. |

The checker accepted no multiplier under its strict rule that all 20 images must have a detected face and no configured headwear or earring terms. Its term counts also disagree with the seed-0 visual review: it sometimes stops mentioning attributes that are still plainly visible. The automated counts therefore cannot support a claim that the attribute was removed.

## Takeaway for the presentation

- **Q1:** Feature 138 appears to change headwear styling in the checked example, rather than remove a headscarf. Feature 3725 appears to affect facial adornment or jewelry as well as other appearance details; the checked example does not support calling it a headscarf-removal feature.
- **Q2:** Neither pilot establishes a usable mitigation threshold. Faces remained recognizable in the reviewed example, but the target headwear remained too. The apparent rise in checker passes for feature 3725 is not reliable because visible jewelry/headwear persisted.
- **Q3:** The next useful experiment is a manually reviewed set with verified headscarf-present and headscarf-absent images, across several seeds and prompt groups. Score face preservation and headscarf presence separately, then compare the first multiplier that changes each outcome across images. Keep the checker’s errors in the report.

## Artifacts and limits

- Feature 138 results: [`candidate_summary.csv`](feature_138/run_biased_002/candidate_screen/candidate_summary.csv), [`REPORT.md`](feature_138/run_biased_002/candidate_screen/REPORT.md), and [`seed 0 grid`](feature_138/run_biased_002/candidate_screen/grids/middle_eastern_woman_seed000.jpg).
- Feature 3725 results: [`candidate_summary.csv`](feature_3725/run_biased_slurm_4109210_1/headwear_screen/candidate_summary.csv), [`REPORT.md`](feature_3725/run_biased_slurm_4109210_1/headwear_screen/REPORT.md), and [`seed 0 grid`](feature_3725/run_biased_slurm_4109210_1/headwear_screen/grids/middle_eastern_woman_seed000.jpg).
- The 20 images per feature come from one prompt group. The visual checks here inspect seed 0 only. These results are exploratory and do not establish a general threshold or bias-mitigation effect.
