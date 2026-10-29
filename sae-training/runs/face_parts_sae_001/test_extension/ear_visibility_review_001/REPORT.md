# Ear visibility and ear-feature check

## Visual review

A balanced sample of 120 generated portraits was reviewed: 10 seeds in each of the 12 race/gender prompt cells. Manual labels record visible ears, camera angle, and head covering. Counts: **47 with visible ears, 25 partly visible, 48 not visible**. Camera angle: **80 frontal, 35 three-quarter, 5 profile**. These are generated-prompt categories, not verified real identities.

## Does camera angle explain missing ears?

No. Of 80 frontal portraits, 38 had no visible ear; nearly all of those had a head covering or hair obscuring the ears. The head-covering group had no visible ears in 37/42 images, compared with 11/78 images without a head covering. Long hair, especially in the white-woman prompt group, also hid ears. Three-quarter/profile views increased ear visibility, but many frontal faces still showed ears and many frontal faces did not. So pose is one factor; hair and head coverings are strong additional factors.

The automatic ear-box detector found boxes in only **7/120** images: 4 visibly present ears and 3 partial ears. It missed **43/47** clear visible ears and **22/25** partial ears. Thus most missing ear-region activations in this run cannot be interpreted as the model failing to represent an ear; the region detector usually did not provide an ear box to evaluate. An absent detector box is not evidence that the ear itself is absent.

## Feature 4858

Feature 4858 was selected on validation data as the candidate most spatially associated with ear boxes for the face-parts SAE. The saved epoch-20 SAE was re-encoded on these images and feature 4858 was pooled only inside usable automatic ear boxes (box area <=12% of image; at least two latent cells). This yielded only 6 usable images: mean activation 14.42 for 3 visible-ear samples and 12.30 for 3 partial-ear samples; there were no usable boxes among images labeled no-visible-ear. This sample is far too small to say whether feature 4858 responds specifically to ears.

## Files

- `manual_annotations.csv`: visual ear, pose, and covering labels for all 120 images.
- `ear_activation_review.csv`: labels, detector-box counts, and feature 4858 ear-region activations.
- `group_visibility.csv` and `summary.json`: counts by prompt group, pose, and detector presence.
- `ear_feature_4858_jet.png`: JET plot of feature 4858 in detected ear boxes; blank columns are images without usable boxes.
- The 12 larger review sheets and `annotation_template.csv` are included for inspection.

These findings are exploratory. The contact-sheet annotations are a manual visual review, not an independent expert annotation study.
