# SAE pilot results

Trained a Top-K SAE on residual activations from SDXL Turbo's
`unet.down_blocks.2.attentions.1`, using 10 varied prompts and four seeds per
prompt. Two additional prompts were held out. The dataset contains 10,240
training vectors and 2,048 held-out vectors, each with 1,280 channels.
The dictionary has 5,120 features with at most 20 active per vector.

After 500 Adam steps, training batch MSE decreased from 7.774 to 0.856.
Held-out normalized MSE decreased from 1.221 to 0.976, where 1 corresponds
to predicting the held-out set's channel-wise mean. Thus generalization is
still weak. 86.6% of dictionary features never activated on this small held-out
sample; this is not a measurement of global dead-feature rate.

For `a goose swimming on a lake at sunrise`, seed 42, generated an unmodified
image, an SAE-only residual reconstruction, and separate sweeps of features
3677, 3123, and 3432 at strengths 0, 0.5, 1, and 2.
The edits preserve the residual reconstruction error. Every strength-1 image
was verified pixel-identical to the original. Hook-level identity, removal,
and doubling checks also passed.

The grid shows changes in lighting, reflection, and the bird's shape.
In this image, reducing feature 3432 changes the sun and reflection; doubling
it darkens that area. This is an observation from one image, not an established
semantic feature label. The comparison prompt was included in training.

No image-quality improvement has been established. A next experiment should
use substantially more varied prompts, address unused features, and evaluate
interventions on multiple unseen prompts and seeds using human preference or
a separately chosen image-quality measure. The existing script can collect
more data through `--prompts` and `--seeds`, and reuse the trained SAE through
`--checkpoint` without retraining.

Files: `comparison.png`, individual images, `activations.pt`, `sae/`,
`settings.json`, `prompts.json`, `features.json`, `training_metrics.json`,
and `image_changes.json`.
