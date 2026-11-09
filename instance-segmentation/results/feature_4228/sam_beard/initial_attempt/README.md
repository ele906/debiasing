# Initial SAM beard-mask attempt (not validated)

This folder preserves the first prompt-guided SAM attempt on
`indian_man/seed053_alpha_p7.png` from Feature 4228.

## Result

The overlay shows that SAM selected a large portion of the face rather than
only the beard. The recorded 60,748 mask pixels and 0.9297 predicted-IoU score
are **not beard-area measurements**: SAM's score measures agreement with its
own prompted object proposal, not semantic beard accuracy.

This is a failure baseline for prompt refinement. Do not use it for an
across-alpha comparison. The saved assets are:

- `*_beard_overlay.png`: red overlay showing the over-segmented proposal.
- `*_beard_mask.png`: corresponding binary mask.
- `*_metrics.txt`: source image, pixel count, and SAM's predicted score.

The runner at `instance-segmentation/segment_sam_feature4228_beard.py` is the
starting point for further manual prompt refinement and visual review.
