"""Validate saved fixed-value latents and render paired originals without semantic claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run import csv_write,atomic_json,digest,inventory

p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
cfg=json.loads((run/'config/experiment_config.json').read_text());workers=json.loads((run/'config/parallel_workers.json').read_text())
allrows={};attempts=[];complete_workers=0
for w in workers:
 worker=Path(w['run']);latest={}
 for line in (worker/'logs/events.jsonl').read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  attempts.append(dict(r,worker=str(worker)));latest[r['image_id']]=r
 for r in latest.values():
  for key in ('image_path','activation_path'):r[key]=str((worker/r[key]).relative_to(run))
  allrows[r['image_id']]=r
 complete_workers+=int((worker/'results/generation_complete.json').exists())
rows=list(allrows.values());csv_write(run/'results/all_generations.csv',rows);csv_write(run/'results/all_attempts.csv',attempts)
baselines={r['prompt_id']:r for r in rows if r['intervention_type']=='baseline' and r['generation_status']=='complete'}
measures=[]
for r in rows:
 if r['intervention_type']!='fixed_activation' or r['generation_status']!='complete':continue
 with np.load(run/r['activation_path']) as z:
  before=z['original_0'][...,0];after=z['modified_0'][...,0];post=z['post_reencoded_0'][...,0]
  assert np.all(after==r['activation_target']),r['image_id']
  metric=dict(r,original_nonzero_fraction=float(np.mean(before!=0)),modified_nonzero_fraction=float(np.mean(after!=0)),
      selected_negative_fraction=float(np.mean(after<0)),introduced_outside_original_support_fraction=float(np.mean((before==0)&(after!=0))),
      post_reencoded_nonzero_fraction=float(np.mean(post!=0)),post_reencoded_vs_set_selected_l2=float(np.linalg.norm(post-after)))
  if 'full_original_0' in z:
   orig=z['full_original_0'];modified=z['full_modified_0']
   metric.update(original_mean_support=float(np.mean(np.count_nonzero(orig,axis=-1))),
    modified_mean_support=float(np.mean(np.count_nonzero(modified,axis=-1))),
    relative_full_latent_change_l2=float(np.linalg.norm(modified-orig)/max(np.linalg.norm(orig),1e-12)))
 with Image.open(run/r['image_path']) as im,Image.open(run/baselines[r['prompt_id']]['image_path']) as base:
  metric['pixel_mae_vs_baseline']=float(np.abs(np.asarray(im).astype(float)-np.asarray(base).astype(float)).mean())
  if r['activation_target']==0 and r['original_activation_zero']:
   assert np.array_equal(np.asarray(im),np.asarray(base)),'Inactive zero target must be no-op'
 measures.append(metric)
csv_write(run/'results/fixed_activation_measurements.csv',measures)
for group in ('black_man','east_asian_man'):
 targets=cfg['set_values'];sheet=Image.new('RGB',(256*(len(targets)+1),310*5),'white');draw=ImageDraw.Draw(sheet)
 for i,seed in enumerate(range(30,35)):
  prompt=f'{group}_seed{seed:03d}';draw.text((5,i*310+1),prompt,fill='black')
  base=baselines.get(prompt)
  if base:
   with Image.open(run/base['image_path']) as im:
    tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(0,i*310+45))
  draw.text((5,i*310+20),'No intervention',fill='black')
  for j,target in enumerate(targets,1):
   match=[r for r in measures if r['prompt_id']==prompt and r['activation_target']==target]
   draw.text((j*256+5,i*310+20),f'Set z4228 = {target:g}',fill='black')
   if not match:
    draw.text((j*256+5,i*310+70),'Pending / failed: see manifest',fill='black');continue
   with Image.open(run/match[0]['image_path']) as im:
    tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(j*256,i*310+45))
 sheet.save(run/'figures'/f'{group}_signed_targets.png')
atomic_json(run/'results/validation.json',dict(expected_edited_images=130,completed_edited_images=len(measures),
 completed_workers=complete_workers,all_saved_modified_values_exact=True,
 note='All available complete fixed-value arrays were checked. Semantic beard appearance is not inferred from tensor activation or pixel changes. Equal coordinate targets are not equal perturbations in active and inactive contexts.'))
(run/'results/research_notes.md').write_text('This experiment tests whether a hypothesized beard feature changes visible facial hair when set uniformly across spatial cells. It does not establish a validated beard measurement or demographic causal specificity. Negative targets violate encoder nonnegativity; uniform nonzero targets often exceed Top-K support. Naturalness measurements and actual perturbation norms are preserved separately. Original PNGs are untouched; contact sheets are derivatives.\n')
inventory(run)
print(json.dumps(dict(completed=len(measures),expected=130,workers_complete=complete_workers)))
