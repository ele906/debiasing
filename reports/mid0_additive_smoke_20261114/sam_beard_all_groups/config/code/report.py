"""Diagnostics and full-group galleries; no semantic accuracy claims without references."""
import argparse,csv,itertools,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,sha,save_json,csv_write

def mask_iou(a,b):
 union=np.logical_or(a,b).sum();return float(np.logical_and(a,b).sum()/union) if union else 1.

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();cfg=json.loads((run/'config/experiment_config.json').read_text())
 images=list(csv.DictReader((run/'config/images.csv').open()));latest={}
 for line in (run/'logs/events.jsonl').read_text().splitlines():
  try:r=json.loads(line);latest[(r['image_id'],r['prompt_set'])]=r
  except ValueError:pass
 results=[];all_candidates=[];comparisons=[]
 for image in images:
  modes=[]
  for mode in cfg['prompt_sets']:
   r=latest.get((image['image_id'],mode),dict(image_id=image['image_id'],group=image['group'],condition=image['condition'],prompt_set=mode,status='not_started'))
   results.append(r)
   if r['status']=='complete':
    assert sha(run/r['array_path'])==r['array_sha256']
    with np.load(run/r['array_path']) as d:
     assert d['candidate_masks'].shape==(3,512,512);assert d['low_res_logits'].shape==(3,256,256)
    for c in r['candidates']:all_candidates.append(dict(image_id=r['image_id'],group=r['group'],condition=r['condition'],prompt_set=mode,**c))
    with Image.open(run/r['selected_mask_path']) as im:modes.append(np.asarray(im)>0)
  if len(modes)==2:comparisons.append(dict(image_id=image['image_id'],group=image['group'],condition=image['condition'],selected_mask_iou=mask_iou(*modes),meaning='Prompt sensitivity, not mask accuracy'))
 csv_write(run/'results/proposals.csv',[dict(r,candidates=json.dumps(r.get('candidates'))) for r in results]);csv_write(run/'results/all_candidate_metrics.csv',all_candidates);csv_write(run/'results/prompt_sensitivity.csv',comparisons)
 paired=[]
 for group,mode in itertools.product(cfg['groups'],cfg['prompt_sets']):
  base=next(r for r in results if r['group']==group and r['condition']=='baseline' and r['prompt_set']==mode)
  for condition in ('fixed_10','fixed_50'):
   r=next(r for r in results if r['group']==group and r['condition']==condition and r['prompt_set']==mode)
   ok=base['status']=='complete' and r['status']=='complete'
   paired.append(dict(group=group,prompt_set=mode,condition=condition,baseline_status=base['status'],edited_status=r['status'],
    proposal_coverage_change_pp=100*(r['roi_coverage']-base['roi_coverage']) if ok else None,
    proposal_area_change_px=r['mask_pixels']-base['mask_pixels'] if ok else None,
    semantic_validation='unverified; not accepted beard coverage'))
 csv_write(run/'results/paired_proposal_changes.csv',paired)
 # All candidates and all inputs remain; visual sheets use the predeclared highest-quality candidate.
 headers=['Saved original','Frozen baseline prompts','Box only: selected proposal','Box + points: selected proposal']
 for group in cfg['groups']:
  sheet=Image.new('RGB',(4*256,3*304+55),'white');draw=ImageDraw.Draw(sheet)
  draw.text((5,3),group+' | seed30 | raw masks, not validated beard labels',fill='black')
  for j,title in enumerate(headers):draw.text((j*256+4,28),title,fill='black')
  for i,condition in enumerate(('baseline','fixed_10','fixed_50')):
   image=next(r for r in images if r['group']==group and r['condition']==condition)
   draw.text((5,55+i*304),condition,fill='black')
   r=latest.get((image['image_id'],'box_only'),{});q=latest.get((image['image_id'],'box_plus_points'),{})
   paths=[run/image['image_path'],run/r['prompt_view_path'] if r.get('prompt_view_path') else None,
          run/r['overlay_path'] if r.get('overlay_path') else None,run/q['overlay_path'] if q.get('overlay_path') else None]
   for j,path in enumerate(paths):
    if path and path.exists():
     with Image.open(path) as im:tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(j*256,55+i*304+24))
    else:draw.text((j*256+5,55+i*304+50),'Unavailable: see status CSV',fill='black')
  sheet.save(run/'figures'/(group+'.png'))
 # Baseline overview by prompt gender for transparent failure inspection.
 for gender in ('man','woman'):
  groups=[g for g in cfg['groups'] if g.endswith('_'+gender)];sheet=Image.new('RGB',(3*256,len(groups)*300+50),'white');draw=ImageDraw.Draw(sheet)
  for j,title in enumerate(('Baseline original','Box only proposal','Box + points proposal')):draw.text((j*256+5,10),title,fill='black')
  for i,group in enumerate(groups):
   image=next(r for r in images if r['group']==group and r['condition']=='baseline')
   draw.text((5,50+i*300),group,fill='black')
   paths=[run/image['image_path']]
   for mode in cfg['prompt_sets']:
    r=latest.get((image['image_id'],mode),{});paths.append(run/r['overlay_path'] if r.get('overlay_path') else None)
   for j,path in enumerate(paths):
    if path and path.exists():
     with Image.open(path) as im:tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(j*256,50+i*300+24))
  sheet.save(run/'figures'/('baseline_overview_'+gender+'.png'))
 complete=[r for r in results if r['status']=='complete'];n_candidates=len(all_candidates)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for mode,marker in zip(cfg['prompt_sets'],('o','s')):
  r=[r for r in complete if r['prompt_set']==mode]
  axes[0].scatter([r['sam_quality_score'] for r in r],[r['roi_coverage'] for r in r],marker=marker,label=mode,alpha=.6)
  axes[1].scatter([r['outside_roi_fraction'] for r in r],[r['lip_coverage'] for r in r],marker=marker,label=mode,alpha=.6)
 axes[0].set(xlabel='SAM model-estimated mask quality',ylabel='Unvalidated proposal coverage in fixed lower-face ROI',title='Quality score does not identify beard')
 axes[1].set(xlabel='Raw proposal fraction outside lower-face ROI',ylabel='Fraction of lip polygon included',title='Geometric diagnostics, not semantic accuracy')
 for ax in axes:ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(run/'figures/prompt_diagnostics.png',dpi=300);fig.savefig(run/'figures/prompt_diagnostics.pdf');plt.close(fig)
 validation=dict(n_images=len(images),n_groups=len(cfg['groups']),expected_predictions=72,complete_predictions=len(complete),candidate_masks=n_candidates,
    image_conditions=['baseline','add10','add50'],all_originals_hash_verified=all(sha(run/r['image_path'])==r['image_sha256'] for r in images),
    all_masks_original_resolution=True,n_generated_diffusion_images=0,n_trained_models=0,semantic_accuracy_measured=False)
 save_json(run/'results/validation.json',validation)
 # Optional review fields remain blank; gender/prompt category is not a beard-presence label.
 if not (run/'results/qualitative_checklist.csv').exists():
  csv_write(run/'results/qualitative_checklist.csv',[dict(image_id=r['image_id'],group=r['group'],condition=r['condition'],prompt_set=r['prompt_set'],
     visible_facial_hair='',mask_follows_hair='',selects_skin_or_lips='',selects_neck_or_clothing='',misses_fine_hair='',acceptable_for_area_measurement='',notes='') for r in results])
 report=['# SAM ViT-B pilot across all race/gender prompt groups','',
    '**Existing images only; no new diffusion generation and no training.** This is a segmentation feasibility pilot, not a validated beard-area experiment.','',
    f'Included: {len(images)} saved feature-4228 images, all12 prompt groups, seed30, and baseline/+10/+50 conditions. Completed SAM prompt predictions: {len(complete)}/72; all{n_candidates} candidate masks and scores are retained.','',
    '## Prompt policy','',
    'The earlier Indian-image-specific coordinates have been replaced by one anatomical rule for every group. A baseline FaceLandmarker detection defines a box from the lower lip region to an extended chin region, with lateral face margins. Five positive points hypothesize lower-jaw/chin foreground; three negative points mark mouth and upper-cheek locations. The same baseline box/points are reused unchanged for the paired edits. Box-only and box-plus-points use the same box. No race/gender-specific thresholds, prompts, or presumed beard labels are used.','',
    '**Important:** positive anatomical points may land on skin or neck in a clean-shaven image. They are geometric hypotheses, not annotated beard pixels. Even box-only SAM can segment a non-beard object. A nonempty proposal is not beard detection, and the highest SAM quality score is not the probability of a beard.','',
    '## Baseline checks across all groups','',
    '![All man-prompt baselines](figures/baseline_overview_man.png)','',
    '![All woman-prompt baselines](figures/baseline_overview_woman.png)','',
    'Gender labels are prompt conditions, not assumptions about the presence or absence of facial hair.','',
    '## Paired images and masks by group','']
 for group in cfg['groups']:report +=[f'### {group}','',f'![{group} originals and selected masks](figures/{group}.png)','']
 report +=['## Measurements and limits','',
    'Raw mask area, lower-face ROI overlap, lip overlap, candidate agreement and between-prompt IoU are saved for diagnosis. Paired changes use the same baseline ROI denominator. These are **SAM-proposal changes**, not accepted changes in beard coverage. The geometric ROI extends below the chin and may include skin/neck. It is not semantic ground truth.','',
    'The top candidate is selected by SAM quality score before looking at conditions; all alternatives remain available. High agreement or model confidence can be confidently wrong about what object was segmented. Failed detections/segmentations remain in status tables and are not silently excluded. There is one seed per group, so no group-level segmentation-accuracy or fairness claims are made.','',
    'Inspect whether the selected regions follow facial hair, especially in images without an obvious beard. If substantial non-hair regions are selected, do not score all4,200 images as beard area. SAM may remain useful as a prompt-assisted mask tool for visible beards, but would need a separate reliable presence gate or reviewed masks before quantitative semantic use.','',
    '## Preservation and reproduction','',
    'originals/: untouched512px PNG hardlinks. landmarks/: full landmarks, fixed soft ROI and lips polygon. config/frozen_prompts.json: all boxes, points and labels. masks/: every candidate and selected mask at512px. sam_arrays/: lossless candidate masks, quality scores and low-resolution logits. image_embeddings/: cached SAM embeddings. overlays/ and figures/: separate derivatives. logs/events.jsonl: append-only attempts. results/: per-candidate and selected-proposal diagnostics.','',
    'The selected diffusion/SAE image provenance is recorded with SHA-256 '+cfg['sae_sha256']+'. SAM ViT-B weights and the FaceLandmarker task are separately hashed. Frozen SAM inference runs in float32; no SAE or SDXL model is loaded.','',
    'To resume: submit instance-segmentation/all_group_sam/pilot.sbatch RUN_DIRECTORY. To rebuild this report: run report.py --run RUN_DIRECTORY in the recorded NumPy/Pillow/Matplotlib environment. Do not rerun prepare.py on an existing directory; it creates a new timestamped pilot.','']
 (run/'README.md').write_text('\n'.join(report))
 for p in Path(__file__).parent.glob('*'):
  if p.suffix in ('.py','.sbatch'):
   dst=run/'config/code'/p.name
   if not dst.exists():shutil.copy2(p,dst)
 manifest=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.csv' and p.name!='run.lock']
 csv_write(run/'output_manifest.csv',manifest)
 print(json.dumps(validation))
if __name__=='__main__':main()
