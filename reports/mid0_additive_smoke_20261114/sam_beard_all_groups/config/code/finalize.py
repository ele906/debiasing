"""Freeze qualitative pilot findings and all-candidate baseline previews; no new inference."""
import csv,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,sha,csv_write,save_json


def tint(rgb,mask):
 out=rgb.astype(float).copy();out[mask]=.52*out[mask]+.48*np.array([255,70,0]);return np.uint8(np.rint(out))

def main():
 run=Path(sys.argv[1]).resolve();cfg=json.loads((run/'config/experiment_config.json').read_text());rows={}
 for line in (run/'logs/events.jsonl').read_text().splitlines():
  try:r=json.loads(line);rows[(r['image_id'],r['prompt_set'])]=r
  except ValueError:pass
 images=list(csv.DictReader((run/'config/images.csv').open()))
 for group in cfg['groups']:
  image=next(r for r in images if r['group']==group and r['condition']=='baseline')
  with Image.open(run/image['image_path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
  sheet=Image.new('RGB',(4*256,2*306+40),'white');draw=ImageDraw.Draw(sheet);draw.text((5,5),group+' baseline: all SAM alternatives, not accepted beard labels',fill='black')
  for i,mode in enumerate(cfg['prompt_sets']):
   r=rows[(image['image_id'],mode)];draw.text((5,40+i*306),mode+'; first column = saved original',fill='black')
   with np.load(run/r['array_path']) as a:masks=a['candidate_masks'];scores=a['candidate_quality_scores']
   tiles=[Image.fromarray(rgb)]+[Image.fromarray(tint(rgb,m)) for m in masks]
   for j,tile in enumerate(tiles):
    tile.thumbnail((256,256));sheet.paste(tile,(j*256,40+i*306+36))
    if j:draw.text((j*256+4,40+i*306+15),f'Candidate{j-1} quality={scores[j-1]:.3f}'+(' SELECTED' if j-1==r['selected_candidate'] else ''),fill='black')
  sheet.save(run/'figures'/('candidates_'+group+'.png'))
 observations=[
  dict(image_id='indian_man_seed030_baseline',prompt_set='box_plus_points',observation='Proposal captures much of the visible white beard, with boundaries still unvalidated.'),
  dict(image_id='indian_man_seed030_baseline',prompt_set='box_only',observation='Top proposal selects a small clothing/neck-side region rather than the visible beard.'),
  dict(image_id='white_woman_seed030_baseline',prompt_set='box_plus_points',observation='On this displayed face without an obvious beard, the proposal includes chin/neck skin.'),
  dict(image_id='hispanic_woman_seed030_baseline',prompt_set='box_only',observation='Proposal includes jewelry/necklace rather than a beard region.'),
  dict(image_id='middle_eastern_woman_seed030_baseline',prompt_set='box_only',observation='Proposal includes head-covering fabric below the face rather than a beard region.')]
 for r in observations:
  proposal=rows[(r['image_id'],r['prompt_set'])];r.update(source='Assistant visual inspection of displayed overlays; qualitative, not human ground truth or measured accuracy',sam_quality_score=proposal['sam_quality_score'],mask_pixels=proposal['mask_pixels'])
 csv_write(run/'results/illustrative_visual_observations.csv',observations)
 section=['## Pilot decision: not ready for automatic beard-area scoring','',
   'All72 prompt evaluations returned nonempty masks. Inspection of the complete baseline overviews shows that several proposals follow neck/chin skin, lips, jewelry or head-covering fabric rather than visible beard hair. This is a qualitative failure-mode diagnosis, not a measured false-positive rate: no pixel ground truth or formal beard-presence labels were supplied.','',
   'SAM remains useful as a prompt-assisted region segmenter. For example, the box-plus-points Indian-man proposal captures much of the visible white beard (quality0.934), but a white-woman chin/neck-skin proposal receives quality0.955. The model-quality score therefore cannot be used here as a beard-presence threshold. The box-only Indian-man proposal also selects a non-beard region, illustrating prompt dependence.','',
   'The observed failures concern this fixed prompt policy, not an estimated race/gender performance disparity. There is one seed per group, and visible facial-hair presence is not inferred from prompt gender. All twelve groups and all three source conditions are retained.','',
   '**Do not run all4,200 images through this policy and label the result beard coverage.** A beard-presence gate or explicitly hair-grounded prompts must first be checked on diverse examples. Clipping masks to the lower-face ROI removes some geometric leakage but does not distinguish skin from hair. Candidate agreement and quality scores cannot substitute for that semantic check.','',
   'A reasonable next step is a small, independently checked facial-hair presence/attribute scorer, with SAM retained for outlining confirmed visible hair. Alternatively, use SAM as an assistive mask tool with a few actual hair/background points on a reviewed subset. Neither additional training nor a new diffusion run was performed in this pilot.','',
   '### All-candidate baseline previews','']
 for group in cfg['groups']:section +=[f'[{group}: all three candidates per prompt](figures/candidates_{group}.png)','']
 readme=run/'README.md';text=readme.read_text();prefix='\n'.join(section)+'\n'
 if '## Pilot decision:' not in text:readme.write_text(text.replace('## Prompt policy',prefix+'## Prompt policy',1))
 # Preserve the frozen code actually used; add later reporting code separately.
 for source in Path(__file__).parent.glob('*'):
  if source.suffix in ('.py','.sbatch'):
   dst=run/'config/code'/source.name
   if not dst.exists():shutil.copy2(source,dst)
 sam_sources=[ROOT/'segment-anything/segment_anything'/p for p in ('predictor.py','build_sam.py','modeling/sam.py','modeling/mask_decoder.py')]
 csv_write(run/'config/sam_source_hashes.csv',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sam_sources])
 manifest=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('output_manifest.csv','run.lock')]
 csv_write(run/'output_manifest.csv',manifest)
 print('All12 groups, 36 images, 72 predictions, 216 masks preserved. Pilot decision: semantic area metric not validated.')
if __name__=='__main__':main()
