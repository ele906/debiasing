# SAM prompt probe on a beard-focused crop

Input: `/n/fs/goose/instance-segmentation/results/feature_4228/beard_annotation_pilot/images/indian_man__seed030__alpha_p7.png`. SAM ViT-B was run with two manually chosen prompt sets. Red overlays show SAM's top-scoring proposal for each set. This is a qualitative prompt comparison only: there is no hand-labeled reference mask, so the SAM predicted IoU is not beard accuracy. Review the overlays before considering any mask as a beard result.

- [Prompt comparison](prompt_comparison.png)
- [Prompt metrics](prompt_metrics.csv)
