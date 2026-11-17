"""Frozen face-parser inference on deduplicated existing PNGs, with journal/resume."""
import argparse,csv,datetime as dt,fcntl,hashlib,importlib.metadata as md,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import parser_helpers as helpers
from prepare import sha,write

def atomic_json(p,r):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False));os.replace(tmp,p)
def validate(run,r):
 if r.get('status')!='complete':return False
 for field in ['beard_mask']+(['probabilities'] if r.get('probability_available') else []):
  p=run/r[field+'_path']
  if not p.exists() or sha(p)!=r[field+'_sha256']:raise RuntimeError('Saved prediction changed: '+str(p))
 return True
def predict(model,image,device):
 with torch.inference_mode():prob=model(helpers.preprocess(image,device))['probabilities'][0];mask=prob.argmax(0).cpu().numpy().astype(np.uint8);bp=prob[14].cpu().numpy().astype(np.float32);means=prob.mean((1,2)).cpu().numpy()
 return mask,bp,means

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int);a=p.parse_args();run=a.run.resolve();cfg=json.loads((run/'config/experiment_config.json').read_text());a.shards=a.shards or cfg['n_shards'];assert a.shards==cfg['n_shards']
 for path,h in zip(cfg['checkpoint_paths'],cfg['checkpoint_hashes']):assert sha(path)==h
 assets=list(csv.DictReader((run/'config/assets.csv').open()));conditions=list(csv.DictReader((run/'config/conditions.csv').open()))
 torch.set_num_threads(4);device=torch.device('cuda');assert torch.cuda.is_available();torch.backends.cudnn.benchmark=False
 if not a.smoke:
  smoke=json.loads((run/'smoke/validation.json').read_text());assert smoke['passed'] and smoke['checkpoint_hashes']==cfg['checkpoint_hashes']
 log=run/'logs'/('smoke.jsonl' if a.smoke else f'shard{a.shard}.jsonl')
 def event(r):
  with log.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
 with (run/'logs'/('smoke.lock' if a.smoke else f'shard{a.shard}.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  model=helpers.load_model(Path(cfg['parser_repo']),device);model.requires_grad_(False)
  runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pillow=md.version('Pillow'),device=torch.cuda.get_device_name(),precision='float32',torch_cuda=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,checkpoint_hashes=cfg['checkpoint_hashes'])
  atomic_json(run/'config'/('smoke_runtime.json' if a.smoke else f'runtime_shard{a.shard}.json'),runtime)
  if a.smoke:
   selected=[]
   for group in ['indian_man','middle_eastern_man','east_asian_man','black_woman']:
    for alpha in [-7,1,7]:selected.append(next(r for r in conditions if r['method']=='multiplicative' and r['prompt_group']==group and r['seed']=='30' and float(r['multiplier'])==alpha))
   checks=[]
   for c in selected:
    h=c['asset_id'];old=json.loads((run/'assets'/(h+'.json')).read_text());assert validate(run,old)
    with Image.open(run/c['image_path']) as im:mask,prob,means=predict(model,im.convert('RGB'),device)
    counts=np.bincount(mask.ravel(),minlength=19)
    with Image.open(run/old['beard_mask_path']) as im:binary=np.asarray(im)>0
    ok=bool(np.array_equal(binary,mask==14) and all(int(counts[k])==old[f'class_{k}_pixels'] for k in range(19)))
    np.savez_compressed(run/'smoke'/(c['condition_id']+'.npz'),class_map=mask,beard_probability=prob,class_mean_probabilities=means)
    checks.append(dict(condition_id=c['condition_id'],asset_id=h,all_class_counts_match=all(int(counts[k])==old[f'class_{k}_pixels'] for k in range(19)),beard_mask_pixels_exact=bool(np.array_equal(binary,mask==14)),passed=ok));print('Reuse parity',c['condition_id'],ok,flush=True)
   write(run/'smoke/checks.csv',checks);passed=all(r['passed'] for r in checks)
   atomic_json(run/'smoke/validation.json',dict(passed=passed,n_checks=len(checks),checkpoint_hashes=cfg['checkpoint_hashes'],rule='Exact binary class14 masks and all19 class counts on12 saved conditions across4 groups'))
   if not passed:raise RuntimeError('Reuse parity failed; do not launch workers')
   return
  failed=0;done=0;skipped=0
  for asset in assets:
   h=asset['asset_id']
   if int(h[:16],16)%a.shards!=a.shard:continue
   cache=run/'assets'/(h+'.json')
   if not cache.exists():
    for other in cfg.get('reuse_runs',[]):
     previous=Path(other);metadata=previous/'assets'/(h+'.json')
     if not metadata.exists():continue
     prior=json.loads(metadata.read_text())
     if not validate(previous,prior):continue
     oldcfg=json.loads((previous/'config/experiment_config.json').read_text());assert oldcfg['checkpoint_hashes']==cfg['checkpoint_hashes'] and oldcfg['parser_commit']==cfg['parser_commit'];assert sha(previous/'config/code/parser_helpers.py')==sha(run/'config/code/parser_helpers.py')
     reused=dict(prior,prediction_origin='reused_exact_input_archive',original_prediction_origin=prior['prediction_origin'],reused_from_run=str(previous))
     for field in ['beard_mask']+(['probabilities'] if prior.get('probability_available') else []):
      source=previous/prior[field+'_path'];dest=run/'assets'/source.name
      if not dest.exists():os.link(source,dest)
      assert sha(dest)==prior[field+'_sha256'];reused[field+'_path']=str(dest.relative_to(run))
     atomic_json(cache,reused);break
   if cache.exists() and validate(run,json.loads(cache.read_text())):skipped+=1;continue
   stem=h+'_'+str(time.time_ns());started=dict(asset_id=h,status='started',timestamp=dt.datetime.now(dt.timezone.utc).isoformat());event(started)
   try:
    source=run/asset['image_path'];assert sha(source)==h
    with Image.open(source) as im:mask,prob,means=predict(model,im.convert('RGB'),device)
    assert mask.shape==(512,512) and np.isfinite(prob).all()
    counts=np.bincount(mask.ravel(),minlength=19);beard=int(counts[14]);face=int(counts[1:15].sum())
    bp=run/'assets'/(stem+'_beard.png');Image.fromarray((mask==14).astype(np.uint8)*255).save(bp)
    pp=run/'assets'/(stem+'.npz');tmp=pp.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,class_map=mask,beard_probability=prob,class_mean_probabilities=means);f.flush();os.fsync(f.fileno())
    os.replace(tmp,pp)
    r=dict(asset_id=h,status='complete',prediction_origin='new_frozen_parser',beard_mask_path=str(bp.relative_to(run)),beard_mask_sha256=sha(bp),probability_available=True,probabilities_path=str(pp.relative_to(run)),probabilities_sha256=sha(pp),full_class_map_available=True,
      beard_pixels=beard,image_pixels=262144,beard_fraction_image=beard/262144,face_classes_1_to_14_pixels=face,beard_fraction_face_classes_1_to_14=beard/face if face else None,
      beard_mean_probability=float(prob.mean()),beard_selected_pixel_mean_probability=float(prob[mask==14].mean()) if beard else None,
      **{f'class_{i}_pixels':int(v) for i,v in enumerate(counts)},**{f'class_{i}_mean_probability':float(v) for i,v in enumerate(means)})
    atomic_json(cache,r);event(dict(asset_id=h,status='complete',prediction_path=str(cache.relative_to(run)),prediction_sha256=sha(cache)));done+=1
    if done%25==0:print('Shard',a.shard,'new',done,'reused',skipped,flush=True)
   except Exception:
    failed+=1;event(dict(asset_id=h,status='failed',error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
  atomic_json(run/'results'/f'shard{a.shard}_complete.json',dict(new=done,reused=skipped,failed=failed))
  if failed:raise RuntimeError(f'{failed} assets failed; see journal, resume missing assets')
  print('Shard complete',a.shard,done,'new',skipped,'reused',flush=True)
if __name__=='__main__':main()
