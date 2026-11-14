# Bindi and earring control pilot: visual verification

## Question

Does the face SAE learn activations associated with a visible bindi or earrings, and
does lowering a candidate feature affect those attributes? This pilot was intended to
make visually verified matched examples before comparing SAE activations.

## What was tried

1. Generated 24 SDXL Turbo text-to-image portraits: six shared seeds crossed with bindi
   prompt on/off and earring prompt on/off. The prompt conditions were kept as similar
   as practical, and SAE activations were saved.
2. Generated a second 24-image set using negative prompts for the off conditions.
3. Started from one shared portrait and used SDXL Turbo image-to-image for four matched
   keep/remove combinations, recording the face-SAE activation maps.

## Visual verification outcome

The first prompt-only pilot did not create clean controls. Several images in the
"bindi off" conditions still show a dot or mark between the eyebrows. Some "earrings
off" images still show small earrings. The bindi-on condition generally produced a
visible red mark, but it also changed clothing, color, or styling in some paired seeds.
These images cannot support a clean attribute-specific comparison.

The negative-prompt rerun produced strong color and texture artifacts, so it is not a
usable control set. In the image-to-image pilot, the requested bindi and earrings
remained visible in the removal conditions; other facial details also shifted. That
pilot does not provide a valid removal comparison either.

I also downloaded an SDXL inpainting checkpoint and tried target masks on six source
portraits. The first mask placement missed the bindi; after correcting the placement,
the generated forehead patch was visibly artificial. A wider mask removed the dot in
one image but changed the surrounding face. Compositing the edit only inside the mask
preserved the outside pixels, but the inpainted regions still had obvious color blobs
and artifacts around the forehead and ears. A crop-aware inpaint attempt also left a
halo over the bindi. These outputs are not suitable SAE controls.

No bindi-versus-no-bindi or earring-versus-no-earring SAE effect is reported from these
pilots. Activations were saved for technical inspection, but the images failed the
necessary visual checks, so comparing their activations would overstate what the data
can answer.

## Conclusion and next experiment

The previous feature 3725 ablation left a visible bindi in place. These additional
checks do not change that conclusion: there is not yet a reliable prompt-generated
positive/negative pair to test whether another SAE feature tracks the mark or earrings.

Next, obtain valid counterfactual images through a higher quality local-edit workflow or
use a manually reviewed dataset with both attribute states within a matched prompt group.
Change the forehead mark and earrings in separate experiments. Visually verify every
pair, including the edit boundaries, before measuring the SAE at the same layer. The
inpainting checkpoint weights are local only and are not included in this repository.

## Contact sheets

- First prompt-only pilot: `pilot_contact_sheet.jpg`.
- Negative-prompt pilot: `../bindi_earring_pilot_002/pilot_contact_sheet.jpg`.
- Same-source image-to-image pilot: `../bindi_earring_img2img_pilot_001/img2img_pilot_contact_sheet.jpg`.
- Masked inpainting attempt: `../inpaint_attribute_pilot_004/inpaint_pilot_contact_sheet.jpg`.
- Crop-aware bindi attempt: `../inpaint_bindi_crop_pilot_001/contact_sheet.jpg`.

Generation, image-to-image, and inpainting scripts are in `sae-training/generate_bindi_earring_control_pilot.py`,
`sae-training/edit_matched_attribute_img2img_pilot.py`,
`sae-training/inpaint_matched_attribute_pilot.py`, and
`sae-training/inpaint_bindi_crop_pilot.py`.
