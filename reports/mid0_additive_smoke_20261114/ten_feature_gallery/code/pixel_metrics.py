"""Resume-safe CPU evaluation of old PNGs only."""
from concurrent.futures import ProcessPoolExecutor
import argparse,itertools,json,os
from pathlib import Path
import numpy as np
from old_gallery import SOURCE,latest,evaluate,table

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
 parent=latest(SOURCE);bases={r['prompt_id']:r for r in parent.values() if r['intervention_type']=='baseline' and r['generation_status']=='complete'}
 cfg=json.loads((run/'config/config.json').read_text());cache=run/'results/pixel_metrics.jsonl';done={}
 if cache.exists():
  for line in cache.read_text().splitlines():
   try:r=json.loads(line);done[r['image_id']]=r
   except ValueError:pass
 tasks=[]
 for f in cfg['features']:
  folder=SOURCE/'workers'/f'feature_{f}'
  for r in latest(folder).values():
   if r['feature_id']==f and r.get('strength_label')!='zero_control':
    assert r['generation_status']=='complete'
    if r['image_id'] not in done:tasks.append((r,folder/r['image_path'],SOURCE/bases[r['prompt_id']]['image_path']))
 with ProcessPoolExecutor(max_workers=4) as pool,cache.open('a') as log:
  for i,r in enumerate(pool.map(evaluate,tasks,chunksize=4)):
   log.write(json.dumps(r)+'\n');log.flush();done[r['image_id']]=r
   if (i+1)%100==0:os.fsync(log.fileno());print('Measured',len(done),'of4200',flush=True)
 assert len(done)==4200
 measured=list(done.values());table(run/'results/per_image_pixel_results.csv',measured);summaries=[]
 for f in cfg['features']:
  for label in ('small','moderate','matched_small','matched_moderate','fixed_10','fixed_20','fixed_50'):
   for activity in ('all','active','inactive'):
    s=[r for r in measured if r['feature_id']==f and r['strength_label']==label and (activity=='all' or (not r['original_activation_zero'])==(activity=='active'))]
    if not s:continue
    vals=[np.mean([r['pixel_mae'] for r in s if r['seed']==seed]) for seed in cfg['seeds'] if any(r['seed']==seed for r in s)]
    v=np.array(vals);ix=np.array(list(itertools.product(range(len(v)),repeat=len(v))));boot=v[ix].mean(1)
    summaries.append(dict(feature=f,strength=label,activity=activity,n_images=len(s),n_seed_clusters=len(v),mean_pixel_mae=float(v.mean()),
     ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),mean_ssim=float(np.mean([r['ssim'] for r in s])),n_exact_noops=sum(r['pixels_identical'] for r in s),mean_applied_l2=float(np.mean([r['decoded_perturbation_norm'] for r in s]))))
 table(run/'results/feature_strength_summary.csv',summaries)
 (run/'results/complete.json').write_text(json.dumps(dict(edited_images=4200,new_generations=0,features=10,seeds=5)))
 print('COMPLETE4200',flush=True)
if __name__=='__main__':main()
