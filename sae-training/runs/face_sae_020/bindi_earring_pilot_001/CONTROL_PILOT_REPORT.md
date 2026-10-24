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

No bindi-versus-no-bindi or earring-versus-no-earring SAE effect is reported from these
pilots. Activations were saved for technical inspection, but the images failed the
necessary visual checks, so comparing their activations would overstate what the data
can answer.

## Conclusion and next experiment

The previous feature 3725 ablation left a visible bindi in place. These additional
checks do not change that conclusion: there is not yet a reliable prompt-generated
positive/negative pair to test whether another SAE feature tracks the mark or earrings.

Next, create same-image counterfactuals with a verified local edit or inpainting model,
changing only the forehead mark or only the earrings while preserving identity, pose,
lighting, and clothing. Visually verify every pair before measuring the SAE at the same
layer. The local model cache currently contains SDXL Turbo and SDXL base, but no dedicated
SDXL inpainting checkpoint. Keep the two attribute tests separate.

## Contact sheets

- First prompt-only pilot: `pilot_contact_sheet.jpg`.
- Negative-prompt pilot: `../bindi_earring_pilot_002/pilot_contact_sheet.jpg`.
- Same-source image-to-image pilot: `../bindi_earring_img2img_pilot_001/img2img_pilot_contact_sheet.jpg`.

Generation and image-to-image scripts are in `sae-training/generate_bindi_earring_control_pilot.py`
and `sae-training/edit_matched_attribute_img2img_pilot.py`.
