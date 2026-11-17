"""Expand frozen scoring grid to all ten existing features; never generate images."""
import argparse,concurrent.futures as cf,csv,datetime as dt,json,os,shutil,subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from prepare import ROOT,BASE,SAE,sha,read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args();ref=a.reference_run.resolve();rcfg=json.loads((ref/'config/experiment_config.json').read_text())
 mult=ROOT/'layers/mid0_stability/combined20_results/per_image.csv';add=BASE/'ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv'
 mrs=read(mult);ars=read(add);assert len(mrs)==69600 and len(ars)==4200
 features=sorted({int(r['feature']) for r in mrs});assert len(features)==10
 run=a.resume.resolve() if a.resume else BASE/('beard_all_features_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 for d in ['config/code','config/source','originals','assets','logs','results','figures','smoke']:(run/d).mkdir(parents=True,exist_ok=True)
 records=[]
 for r in mrs:
  f=int(r['feature']);alpha=float(r['alpha']);records.append(dict(condition_id=f"mult_{f}_{r['group']}_{r['seed']}_{alpha:g}",feature_id=f,method='multiplicative',prompt_group=r['group'],seed=int(r['seed']),strength_label=f'alpha_{alpha:g}',multiplier=alpha,additive_offset='',source_path=r['image'],source_sha256=r['image_sha256'],original_activation_nonzero_fraction=r['active_fraction'],original_activation_zero=r['initially_active']!='True',applied_l2=r['delta_fro_applied'],relative_delta_fro=r['relative_delta_fro'],face_detected=r['face_detected'],source_evaluation=str(mult),negative_multiplier=alpha<0))
 for r in ars:
  records.append(dict(condition_id='add_'+r['image_id'],feature_id=int(r['feature_id']),method='additive',prompt_group=r['group'],seed=int(r['seed']),strength_label=r['strength_label'],multiplier='',additive_offset=float(r['additive_offset']),source_path=r['image_path'],source_sha256=r['image_sha256'],original_activation_nonzero_fraction=r['original_activation_nonzero_fraction'],original_activation_zero=r['original_activation_zero'],applied_l2=r['perturbation_norm'],face_detected='',source_evaluation=str(add),activation_path=r['activation_path'],source_run=r['run_id'],negative_multiplier=False))
 for f,g,s in sorted({(int(r['feature_id']),r['group'],int(r['seed'])) for r in ars}):
  r=next(r for r in ars if int(r['feature_id'])==f and r['group']==g and int(r['seed'])==s);records.append(dict(condition_id=f'baseline_{f}_{g}_{s}',feature_id=f,method='baseline',prompt_group=g,seed=s,strength_label='baseline',multiplier='',additive_offset=0,source_path=r['baseline_image_path'],source_sha256=sha(r['baseline_image_path']),applied_l2=0,negative_multiplier=False))
 assert len(records)==74400
 # Verify every source path/hash in parallel, even when pixels/bytes duplicate another condition.
 expected={r['source_path']:r['source_sha256'] for r in records}
 def verify(item):
  path,h=item;assert sha(path)==h,('Source hash mismatch',path)
  with Image.open(path) as im:assert im.size==(512,512)
  return path,h
 input_identity=dict(mult_sha256=sha(mult),additive_sha256=sha(add),n_paths=len(expected))
 checkpoint=run/'config/verified_inputs.json'
 if checkpoint.exists():assert json.loads(checkpoint.read_text())==input_identity
 else:
  print('Verifying',len(expected),'source files on16 threads',flush=True)
  with cf.ThreadPoolExecutor(max_workers=16) as pool:
   for i,_ in enumerate(pool.map(verify,expected.items()),1):
    if i%5000==0:print('Verified',i,'/',len(expected),flush=True)
  checkpoint.write_text(json.dumps(input_identity,indent=2))
 unique={}
 index={(r['feature_id'],r['prompt_group'],r['seed']):r for r in records if r['method']=='multiplicative' and r['multiplier']==1}
 for i,r in enumerate(records):
  h=r['source_sha256']
  if h not in unique:
   dest=run/'originals'/(h+'.png');
   if not dest.exists():os.link(r['source_path'],dest)
   unique[h]=dict(asset_id=h,image_path=str(dest.relative_to(run)),source_path=r['source_path'],source_sha256=h)
  r.update(asset_id=h,image_path=unique[h]['image_path'],layer='unet.mid_block.attentions.0',sae_sha256=SAE)
  r['control_condition_id']=index[r['feature_id'],r['prompt_group'],r['seed']]['condition_id'] if r['method']=='multiplicative' else f"baseline_{r['feature_id']}_{r['prompt_group']}_{r['seed']}" if r['method']=='additive' else r['condition_id']
 controls=[]
 for r in records:
  if r['method']!='baseline':continue
  m=index[r['feature_id'],r['prompt_group'],r['seed']]
  with Image.open(r['source_path']) as x,Image.open(m['source_path']) as y:equal=np.array_equal(np.asarray(x.convert('RGB')),np.asarray(y.convert('RGB')))
  controls.append(dict(feature_id=r['feature_id'],prompt_group=r['prompt_group'],seed=r['seed'],additive_baseline=r['condition_id'],multiplicative_control=m['condition_id'],pixels_equal=equal))
 write(run/'config/control_equivalence.csv',controls);write(run/'config/conditions.csv',records);write(run/'config/assets.csv',list(unique.values()))
 for p in (ref/'config/code').glob('*'):
  if p.is_file():shutil.copy2(p,run/'config/code'/p.name)
 for p in Path(__file__).parent.iterdir():
  if p.suffix in ['.py','.sbatch']:shutil.copy2(p,run/'config/code'/p.name)
 cfg=dict(rcfg,run_id=run.name,feature_id=None,features=features,n_conditions=len(records),n_unique_assets=len(unique),n_shards=8,reuse_runs=[str(ref)],n_reused_assets='Resolved from reference assets at worker resume',expected_storage_upper_gb=len(unique)*(512*512*5+300000)/1e9)
 assert shutil.disk_usage(run).free>len(unique)*(512*512*5+300000), 'Insufficient space for conservative storage estimate'
 (run/'config/experiment_config.json').write_text(json.dumps(cfg,indent=2))
 for name in ['input_hashes.csv','git_commit.txt','git_status.txt']:shutil.copy2(ref/'config'/name,run/'config'/name)
 for name in ['0_per_image.csv','1_per_image_pixel_results.csv']:shutil.copy2(ref/'config/source'/name,run/'config/source'/name)
 smoke=json.loads((ref/'smoke/validation.json').read_text());assert smoke['passed'];smoke['reused_from']=str(ref/'smoke/validation.json');(run/'smoke/validation.json').write_text(json.dumps(smoke,indent=2))
 settings=[]
 for f in features:
  for label in ['stability_top10_4243081','stability_extra15_4243220']:
   source=ROOT/'layers/runs/mid0_sae_face_001/interventions'/label/f'feature_{f}/settings.json';s=json.loads(source.read_text());assert s['sae_weights_sha256']==SAE and s['block']=='unet.mid_block.attentions.0';shutil.copy2(source,run/'config/source'/f'{label}_{f}.json');settings.append(dict(path=str(source),sha256=sha(source)))
 write(run/'config/feature_settings_hashes.csv',settings)
 print('RUN_DIRECTORY',run,flush=True);print('Unique assets',len(unique),'upperGB',cfg['expected_storage_upper_gb'],flush=True)
if __name__=='__main__':main()
