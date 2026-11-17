"""Exploratory paired SAM proposals; visual screening is separate from inference."""
import argparse,csv,json,hashlib,itertools,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import sha,save_json,csv_write

def load(p):return list(csv.DictReader(p.open()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
 images=load(run/'config/images.csv');cfg=json.loads((run/'config/experiment_config.json').read_text());events={}
 for line in (run/'logs/events.jsonl').read_text().splitlines():
  if line.strip():
   r=json.loads(line);events[r['image_id'],r['prompt_set']]=r
 rows=[];candidates=[]
 for im in images:
  assert sha(run/im['image_path'])==im['image_sha256']
  for mode in cfg['prompt_sets']:
   r=events.get((im['image_id'],mode),dict(image_id=im['image_id'],group=im['group'],prompt_group=im['prompt_group'],seed=int(im['seed']),condition=im['condition'],prompt_set=mode,status='missing'))
   if r['status']=='complete':
    for key in ['array','selected_mask','overlay']:assert sha(run/r[key+'_path'])==r[key+'_sha256']
    with np.load(run/r['array_path']) as d:assert d['candidate_masks'].shape==(3,512,512)
    for c in r['candidates']:candidates.append(dict(image_id=r['image_id'],prompt_group=r['prompt_group'],seed=r['seed'],condition=r['condition'],prompt_set=mode,**c))
   rows.append(r)
 csv_write(run/'results/proposals.csv',[dict(r,candidates=json.dumps(r.get('candidates'))) for r in rows]);csv_write(run/'results/all_candidate_metrics.csv',candidates)
 checks=run/'results/visual_screening.csv'
 if not checks.exists():csv_write(checks,[dict(image_id=im['image_id'],prompt_group=im['prompt_group'],seed=im['seed'],condition=im['condition'],prompt_set=mode,screen_status='unreviewed',note='',reviewer='') for im in images for mode in cfg['prompt_sets']])
 screening={(r['image_id'],r['prompt_set']):r for r in load(checks)}
 # Full paired context sheets, with all inputs and both fixed prompt policies.
 for context in cfg['groups']:
  sheet=Image.new('RGB',(1024,970),'white');draw=ImageDraw.Draw(sheet)
  for j,t in enumerate(['Original','Frozen baseline prompts','Box-only mask','Box + points mask']):draw.text((j*256+3,20),t,fill='black')
  for i,cond in enumerate(['baseline','fixed_10','fixed_50']):
   im=next(x for x in images if x['group']==context and x['condition']==cond);y=55+i*300
   draw.text((4,y),context+' '+cond,fill='black')
   rs=[events.get((im['image_id'],m),{}) for m in cfg['prompt_sets']]
   paths=[run/im['image_path'],run/rs[0]['prompt_view_path'] if 'prompt_view_path' in rs[0] else None]+[run/r['overlay_path'] if 'overlay_path' in r else None for r in rs]
   for j,path in enumerate(paths):
    if path and path.exists():
     with Image.open(path) as pic:pic=pic.convert('RGB');pic.thumbnail((256,256));sheet.paste(pic,(j*256,y+25))
  sheet.save(run/'figures'/(context+'.png'))
 paired=[]
 for group,seed,mode,cond in itertools.product(['indian_man','middle_eastern_man'],range(30,35),cfg['prompt_sets'],['fixed_10','fixed_50']):
  context=f'{group}_seed{seed}';base=events.get((context+'_baseline',mode),{});edit=events.get((context+'_'+cond,mode),{})
  bs=screening.get((context+'_baseline',mode),{}).get('screen_status','unreviewed');es=screening.get((context+'_'+cond,mode),{}).get('screen_status','unreviewed')
  ok=base.get('status')=='complete' and edit.get('status')=='complete'
  paired.append(dict(prompt_group=group,seed=seed,prompt_set=mode,condition=cond,baseline_status=base.get('status','missing'),edit_status=edit.get('status','missing'),baseline_screen=bs,edit_screen=es,
   visually_usable_pair=ok and bs=='usable_proxy' and es=='usable_proxy',
   proposal_coverage_change_pp=100*(edit['roi_coverage']-base['roi_coverage']) if ok else None,
   proposal_area_change_px=edit['mask_pixels']-base['mask_pixels'] if ok else None,
   interpretation='SAM proposal change; semantic membership requires independent validation'))
 csv_write(run/'results/paired_changes.csv',paired)
 summaries=[];rng=np.random.default_rng(4228)
 for group,mode,cond,subset in itertools.product(['indian_man','middle_eastern_man'],cfg['prompt_sets'],['fixed_10','fixed_50'],['all_proposals','visually_screened_proxy']):
  rs=[r for r in paired if r['prompt_group']==group and r['prompt_set']==mode and r['condition']==cond and r['proposal_coverage_change_pp'] is not None and (subset=='all_proposals' or r['visually_usable_pair'])]
  v=np.array([r['proposal_coverage_change_pp'] for r in rs]);n=len(v)
  boots=np.random.default_rng(4228).choice(v,(10000,n),replace=True).mean(1) if n>=2 else np.array([])
  lo,hi=np.percentile(boots,[2.5,97.5]) if len(boots) else (None,None)
  summaries.append(dict(prompt_group=group,prompt_set=mode,condition=cond,subset=subset,n_seed_pairs=n,n_expected=5,mean_change_pp=float(v.mean()) if n else None,ci95_low=lo,ci95_high=hi,ci_method='percentile bootstrap of paired seed deltas; 10000 draws; exploratory; n<=5; no CI if n<2',bootstrap_seed=4228))
 csv_write(run/'results/summary.csv',summaries)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,group in zip(axes,['indian_man','middle_eastern_man']):
  for k,mode in enumerate(cfg['prompt_sets']):
   for seed in range(30,35):
    rs=[next(r for r in paired if r['prompt_group']==group and r['seed']==seed and r['prompt_set']==mode and r['condition']==c) for c in ['fixed_10','fixed_50']]
    vals=[r['proposal_coverage_change_pp'] for r in rs]
    ax.plot(np.array([10,50])+(-1 if k==0 else 1),[np.nan if v is None else v for v in vals],'-o',alpha=.6,color=['#4477AA','#CC6677'][k],label=mode if seed==30 else None)
  ax.axhline(0,color='gray',lw=.8);ax.set_title(group.replace('_',' '));ax.set_xticks([10,50]);ax.set_xlabel('Additive offset');ax.legend(fontsize=8)
 axes[0].set_ylabel('SAM proposal coverage change (percentage points)')
 fig.suptitle('All paired proposals, including failures of semantic interpretation\nEach line is one seed; these are not validated beard-coverage scores',fontsize=11);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(run/'figures'/('paired_proposal_changes.'+ext),dpi=220)
 plt.close(fig)
 save_json(run/'results/validation.json',dict(n_images=len(images),n_groups=2,n_seeds=5,n_expected_predictions=60,n_completed=sum(r.get('status')=='complete' for r in rows),n_candidates=len(candidates),original_hashes_verified=True,n_new_diffusion_images=0,n_trained_models=0,semantics='Visual screening by assistant is not independent ground truth; proposal metrics are exploratory'))
 text='# Exploratory SAM beard-proxy analysis: two prompt groups\n\nExisting feature-4228 images only; no diffusion generation or training. Indian-man and Middle-Eastern-man prompts, seeds30–34, baseline/+10/+50. These groups were selected after viewing the previous pilot, so this is exploratory and cannot establish demographic specificity.\n\n## Measurement\n\nThe same baseline landmark rule defines box/points for each seed, frozen across its edits. Box-plus-points is the primary policy motivated by the previous pilot; box-only is a sensitivity check. Each chooses the highest SAM-quality candidate; all alternatives are saved. Quality is not beard confidence. Coverage = sum(mask × fixed baseline soft lower-face ROI) / sum(ROI). Paired change is edited minus baseline, in percentage points. The ROI can include neck/skin and is not beard ground truth.\n\nAll proposals remain in paired_changes.csv, including semantically unusable masks. Visual screening records usable_proxy/uncertain/non_hair_or_incomplete, with reasons. It is an assistant feasibility check, not independent validation. A screened subset requires both baseline and edit usable; post-intervention exclusions can bias results and remain descriptive. No score in this report establishes a true beard-area change.\n\n## Paired images\n\n'
 text+='## Findings from the complete five-seed check\n\nBox-plus-points overlays broadly follow visible facial hair for all five Middle-Eastern-man seeds at baseline, +10 and +50, with imperfect boundaries. In Indian-man images, only seed30 provides usable paired proxies at both offsets; other masks include substantial skin/neck or are uncertain. This visual screening is by the assistant, not independent ground truth. The group restriction alone therefore does not solve measurement validity.\n\n| Prompt group | Offset | Screened seed pairs | Mean proposal change (pp) | Exploratory 95% bootstrap interval |\n|---|---:|---:|---:|---|\n'
 for r in summaries:
  if r['prompt_set']=='box_plus_points' and r['subset']=='visually_screened_proxy':
   interval='Unavailable (one seed)' if r['ci95_low'] is None else f"[{r['ci95_low']:.2f}, {r['ci95_high']:.2f}]"
   mean='Missing' if r['mean_change_pp'] is None else f"{r['mean_change_pp']:.2f}"
   text+=f"| {r['prompt_group']} | {r['condition']} | {r['n_seed_pairs']}/5 | {mean} | {interval} |\n"
 text+='\nThe +50 Middle-Eastern-man SAM proposal increases modestly; this is an exploratory region-proposal association, not a validated increase in beard hair or proof that feature4228 controls beards. +10 does not give a clear directional change in this proxy. One screened Indian-man seed cannot support a group-level conclusion. Box-only failures show that the measurement depends strongly on prompting. No claim of no semantic effect, demographic causal specificity, or visual naturalness follows.\n\n'
 for context in cfg['groups']:text+=f'### {context}\n\n![Originals and both mask policies](figures/{context}.png)\n\n'
 text+='## Analysis and uncertainty\n\n![All proposal deltas](figures/paired_proposal_changes.png)\n\nresults/summary.csv reports per-group, per-offset paired seed means, sample sizes, and percentile-bootstrap intervals. Five seeds are the units; prompt policies and offsets are not independent observations. No group comparison or confirmatory p-values are claimed. Small-sample intervals and post-selection screening are exploratory. A change in mask area can reflect segmentation drift, skin, pose, or image changes rather than hair.\n\n## Preservation and resume\n\nconfig/code contains the executed source snapshot, configuration and input hashes. originals contains untouched lossless images; overlays/figures are separate derivatives. sam_arrays preserves masks, scores, logits; image_embeddings preserves embeddings. logs/events.jsonl is the incremental journal. Exact matching seed30 images/prompts reuse prior predictions, with provenance. All masks/candidates are retained.\n\n```bash\nsbatch instance-segmentation/two_group_sam/pilot.sbatch '+str(run.resolve())+'\nPYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python '+str(run.resolve())+'/config/code/report.py --run '+str(run.resolve())+'\n```\n\nFull originals/tensors remain on the research host. GitHub contains derived galleries, diagnostic CSVs, configurations and code only.\n'
 (run/'README.md').write_text(text)
 csv_write(run/'output_manifest.csv',[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ['output_manifest.csv','run.lock']])
 print('Report complete',len(rows),'prediction rows',len(candidates),'candidates')
if __name__=='__main__':main()
