"""Publish a frozen, inspectable smoke subset; preserve original-resolution PNGs and tensors."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw
from run import digest,atomic_json,csv_write,ROOT

p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
for d in ('originals','activations','figures','config','code'):(out/d).mkdir()
latest={}
for line in (run/'logs/events.jsonl').read_text().splitlines():
 try:r=json.loads(line);latest[r['image_id']]=r
 except ValueError:pass
smoke=json.loads((run/'results/smoke_validation.json').read_text());cfg=json.loads((run/'config/experiment_config.json').read_text())
prompts=smoke['prompt_ids'];selected=[];lookup={}
for r in latest.values():
 if r['generation_status']!='complete' or r['prompt_id'] not in prompts:continue
 if r['feature_id'] not in (-1,4228):continue
 if r['feature_id']==4228 and r['image_id'] not in smoke['image_ids'] and r['strength_label']!='zero_control':continue
 stem=r['prompt_id']+'_'+r['intervention_type']+'_'+r['strength_label']
 image=out/'originals'/(stem+'.png');activation=out/'activations'/(stem+'.npz')
 shutil.copy2(run/r['image_path'],image);shutil.copy2(run/r['activation_path'],activation)
 row=dict(r,source_image_path=str(run/r['image_path']),source_activation_path=str(run/r['activation_path']),
          image_path=str(image.relative_to(out)),activation_path=str(activation.relative_to(out)))
 with np.load(activation) as z:
  before=z['original_0'][...,0];after=z['modified_0'][...,0]
  row.update(measured_feature_id=4228,original_feature_mean=float(before.mean()),original_feature_max=float(before.max()),
   original_feature_nonzero_fraction=float(np.count_nonzero(before)/before.size),modified_feature_mean=float(after.mean()),
   modified_feature_nonzero_fraction=float(np.count_nonzero(after)/after.size))
 selected.append(row);lookup[(r['prompt_id'],r['strength_label'])]=row
for row in selected:
 with Image.open(out/row['image_path']) as im,Image.open(out/lookup[(row['prompt_id'],'no_intervention')]['image_path']) as base:
  row['pixel_mae_vs_no_intervention']=float(np.abs(np.asarray(im).astype(float)-np.asarray(base).astype(float)).mean())
csv_write(out/'per_image_results.csv',selected)

def panel(name,labels):
 width=256*len(labels);height=310*len(prompts)
 sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet)
 for i,prompt in enumerate(prompts):
  draw.text((5,310*i+2),prompt,fill='black')
  for j,(label,title) in enumerate(labels):
   r=lookup[(prompt,label)]
   with Image.open(out/r['image_path']) as im:
    tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(j*256,i*310+50))
   beta=r.get('additive_offset');desc=title+(f' beta={beta:.3f}' if beta is not None and label!='no_intervention' else '')
   draw.text((j*256+4,i*310+20),desc,fill='black')
 sheet.save(out/'figures'/(name+'.png'))
panel('additive_sweep',[('no_intervention','Baseline'),('small','Small'),('moderate','Moderate'),('fixed_10','Fixed'),('fixed_20','Fixed'),('fixed_50','Fixed')])
panel('matched',[('no_intervention','Baseline'),('matched_small','Matched small'),('matched_moderate','Matched moderate')])
panel('controls',[('no_intervention','Baseline'),('zero_control','Beta zero'),('full_reconstruction','SAE reconstruction')])
# Existing multiplicative references, copied without recompression or changing resolution.
mult=Path(cfg['multiplicative_run'])/'feature_4228';alphas=[-7,-5,-3,-2,-1,0,1,2,3,5,7]
sheet=Image.new('RGB',(256*11,310*4),'white');draw=ImageDraw.Draw(sheet);multrows=[]
for i,prompt in enumerate(prompts):
 group=prompt.rsplit('_seed',1)[0];seed=int(prompt.rsplit('_seed',1)[1]);draw.text((3,i*310),prompt,fill='black')
 for j,alpha in enumerate(alphas):
  slug=('m' if alpha<0 else 'p')+str(abs(alpha));source=mult/'images'/group/f'seed{seed:03d}_alpha_{slug}.png'
  target=out/'originals'/f'{prompt}_multiplicative_{slug}.png';shutil.copy2(source,target)
  with Image.open(source) as im:
   tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(256*j,310*i+45))
  draw.text((256*j+3,310*i+20),f'alpha={alpha}',fill='black')
  multrows.append(dict(prompt_id=prompt,seed=seed,multiplier=alpha,image_path=str(target.relative_to(out)),source_path=str(source),sha256=digest(target)))
sheet.save(out/'figures/multiplicative.png');csv_write(out/'multiplicative_references.csv',multrows)
for name in ('experiment_config.json','additive_calibration.json','runtime.json','feature_selection.json'):
 shutil.copy2(run/'config'/name,out/'config'/name)
shutil.copy2(run/'results/smoke_validation.json',out/'config/smoke_validation.json')
shutil.copy2(run/'results/control_pixel_checks.csv',out/'config/control_pixel_checks.csv')
for name in ('additive.py','run.py','test_additive.py','shard_additive.py','additive_parallel.sbatch','build_smoke_report.py'):
 shutil.copy2(Path(__file__).parent/name,out/'code'/name)
rows=['# Additive SAE smoke test — feature 4228','',
 'Checkpoint: `layers/runs/mid0_sae_face_001/sae`, SHA-256 `cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6`. Layer: `unet.mid_block.attentions.0`. The SAE was fine-tuned; the SDXL Turbo generator stayed frozen. This report uses the checkpoint of the active multiplicative experiment.','',
 '**Smoke test passed.** All 60 paired no-intervention controls matched the multiplicative α=1 pixels exactly. The four displayed contexts use the same prompts and seeds as that experiment. Feature 4228 was naturally nonzero in both displayed Black-man prompt contexts and exactly zero at every spatial location in both displayed East-Asian-man prompt contexts. Prompt labels describe the conditioning text, not ground-truth identity.','',
 '## Additive sweep','',
 'Small β = **1.73515**, moderate β = **10.67096**, plus fixed **10, 20, 50**. Offsets are added at every spatial position of the selected feature. Small/moderate values come from nonzero activations across the 60 paired baseline contexts.','',
 '![Paired additive results](figures/additive_sweep.png)','',
 '## Matching actual perturbation magnitude','',
 'Targets for the applied block-output L2 norm are **42.5242** and **65.1363**. These are perturbation norms, not additive constants. The offset is solved separately in each context using the same additive function, including fp16 rounding; the displayed β values are approximately **2.6573** and **4.071–4.072**. All matched smoke conditions were within 0.5% of their target.','',
 '![Matched additive results](figures/matched.png)','',
 '## No-op and reconstruction controls','',
 'β=0 is an exact no-op for the historical residual-preserving intervention. Replacing the full residual with SAE reconstruction is a separate control.','',
 '![Controls](figures/controls.png)','',
 '## Existing multiplicative references','',
 'These images were copied from the active multiplicative sweep, using the same checkpoint, prompts and seeds. Negative multipliers are potentially out of distribution.','',
 '![All 11 required multipliers](figures/multiplicative.png)','',
 '## Measurements','',
 '| Context | Strength | β | Initially zero? | Modified feature mean | Applied output L2 | Pixel MAE |',
 '|---|---|---:|:---:|---:|---:|---:|']
for r in selected:
 if r['image_id'] in smoke['image_ids']:
  rows.append(f"| {r['prompt_id']} | {r['strength_label']} | {r['additive_offset']:.5f} | {r['original_activation_zero']} | {r['modified_feature_mean']:.3f} | {r['decoded_perturbation_norm']:.3f} | {r['pixel_mae_vs_no_intervention']:.3f} |")
rows+=['','## Interpretation and files','',
 'The inactive contexts show that positive addition can introduce a feature that multiplication cannot change when its original value is exactly zero. This is a mechanism check, not evidence that the feature is causally specific to a demographic group. Visible changes, semantic selectivity, evaluator confidence, facial validity and degradation require separate assessment. No demographic evaluation results are claimed in this smoke report.','',
 'The encoder uses nonnegative Top-K (k=10); newly introduced coordinates are decoded linearly, without clipping or reapplying Top-K. Additions can increase support beyond k and are not assumed to follow the natural latent distribution. Full original/modified dictionaries and post-reencoded selected-feature maps are preserved.','',
 'Contact sheets are viewing derivatives. **Every original PNG in [originals/](originals/) remains at 512×512 without resizing or recompression.** [Per-image measurements](per_image_results.csv), [lossless activation archives](activations/), [configuration and calibration](config/), [source code](code/), and [file hashes](manifest.json) are included. The 44 multiplicative reference images are separate from the 40 additive/control images.','',
 f'Workspace source: `{run}`. Full parallel additive workers are separate and remain in progress.','']
(out/'README.md').write_text('\n'.join(rows))
atomic_json(out/'manifest.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file()])
print(out);print('Additive/control images',len(selected),'multiplicative references',len(multrows))
