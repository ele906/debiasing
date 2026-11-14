# Blinded review packet — feature 1874

There are 48 randomized original-versus-edited pairs: four seeds from each of the 12 prompt categories.  One image in each pair is the unedited original; the other is an intervention at alpha −3, −1, +1, or +3.  The edited side and alpha are hidden.

## How to review

1. Open `review_pages/page_01.jpg` through `page_06.jpg` (eight comparisons per page), or inspect the individual images in `pairs/`.
2. Fill in `review_template.csv` without opening the blind key.  Mark the changed side, whether the effect is mainly in the face, whether image quality degrades, and a short description.
3. When the sheet is complete, open `blind_key_DO_NOT_OPEN_UNTIL_REVIEW_DONE.csv` to score whether the changed side was identified and compare positive vs. negative interventions.

This is a qualitative screening packet, not a claim of semantic interpretation.  The purpose is to decide whether feature 1874 is controlled and face-localized enough to justify an independent held-out sweep.
