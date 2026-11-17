"""Frozen SAM ViT-B inference on pre-existing photos; resumable per image/prompt."""
import argparse,csv,datetime as dt,fcntl,importlib.metadata as md,json,os,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from common import ROOT,sha,save_json,csv_write
sys.path.insert(0,str(ROOT/'segment-anything'))
from segment_anything import SamPredictor,sam_model_registry


def overlay(rgb,mask):
 out=rgb.astype(np.float32).copy();out[mask]=.52*out[mask]+.48*np.array([255,70,0]);return np.uint8(np.rint(out))

def iou(a,b):
 union=int(np.logical_or(a,b).sum());return float(np.logical_and(a,b).sum()/union) if union else 1.


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();run=a.run.resolve()
 cfg=json.loads((run/'config/experiment_config.json').read_text());assert sha(cfg['sam_checkpoint'])==cfg['sam_sha256']
 prompts=json.loads((run/'config/frozen_prompts.json').read_text());images=list(csv.DictReader((run/'config/images.csv').open()))
 cache=run/'logs/events.jsonl';latest={}
 if cache.exists():
  for line in cache.read_text().splitlines():
   try:r=json.loads(line);latest[(r['image_id'],r['prompt_set'])]=r
   except ValueError:pass
 def record(r):
  with cache.open('a') as f:f.write('\n'+json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
 with (run/'logs/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
  model=sam_model_registry['vit_b'](checkpoint=cfg['sam_checkpoint']).to(device).eval();model.requires_grad_(False)
  predictor=SamPredictor(model)
  runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pillow=md.version('Pillow'),device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
    precision='float32',sam_checkpoint_sha256=cfg['sam_sha256'],sam_mask_logit_threshold=float(model.mask_threshold),multimask_output=True,
    candidate_selection='argmax model-estimated mask quality; not semantic beard confidence')
  save_json(run/'config/sam_runtime.json',runtime)
  for image in images[:a.limit] if a.limit else images:
   if image['group'] not in prompts:
    for mode in cfg['prompt_sets']:
     r=dict(image_id=image['image_id'],prompt_set=mode,status='no_baseline_landmarks',group=image['group'],condition=image['condition']);record(r);latest[(r['image_id'],mode)]=r
    continue
   if all(latest.get((image['image_id'],mode),{}).get('status')=='complete' for mode in cfg['prompt_sets']):
    for mode in cfg['prompt_sets']:
     old=latest[(image['image_id'],mode)]
     for field in ('array','selected_mask','overlay'):
      if sha(run/old[field+'_path'])!=old[field+'_sha256']:raise ValueError('Completed artifact changed: '+old[field+'_path'])
    continue
   src=run/image['image_path'];assert sha(src)==image['image_sha256']
   with Image.open(src) as im:rgb=np.asarray(im.convert('RGB')).copy()
   predictor.set_image(rgb)
   emb=run/'image_embeddings'/(image['image_id']+'.npz')
   if not emb.exists():
    temp=emb.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,image_embedding=predictor.get_image_embedding().detach().cpu().numpy(),original_size=np.array(predictor.original_size),input_size=np.array(predictor.input_size));f.flush();os.fsync(f.fileno())
    os.replace(temp,emb)
   prompt=prompts[image['group']]
   annotated=Image.fromarray(rgb);draw=ImageDraw.Draw(annotated);draw.rectangle(prompt['box'],outline=(0,230,255),width=2)
   for point,label in zip(prompt['points'],prompt['labels']):
    x,y=point;draw.ellipse((x-3,y-3,x+3,y+3),fill=(0,255,30) if label else (255,30,30))
   prompt_path=run/'prompt_views'/(image['image_id']+'.png');annotated.save(prompt_path)
   with np.load(run/'landmarks'/(image['group']+'_regions.npz')) as reg:roi=reg['roi'];lips=reg['lips']
   for mode in cfg['prompt_sets']:
    if latest.get((image['image_id'],mode),{}).get('status')=='complete':continue
    attempt=sum(1 for line in cache.read_text().splitlines() if line and '"status": "started"' in line and image['image_id'] in line and mode in line) if cache.exists() else 0
    stem=image['image_id']+'_'+mode+f'_attempt{attempt}'
    r=dict(image_id=image['image_id'],group=image['group'],condition=image['condition'],offset=int(image['offset']),seed=int(image['seed']),prompt_group=image.get('prompt_group',image['group']),prompt_set=mode,
      image_path=image['image_path'],image_sha256=image['image_sha256'],attempt=attempt,status='started',timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
      presence_gate='unverified: no nonempty mask is automatically accepted as beard')
    record(r)
    try:
     points=np.asarray(prompt['points'],np.float32) if mode=='box_plus_points' else None
     labels=np.asarray(prompt['labels'],np.int32) if points is not None else None
     with torch.inference_mode():masks,scores,logits=predictor.predict(point_coords=points,point_labels=labels,box=np.asarray(prompt['box'],np.float32),multimask_output=True)
     best=int(np.argmax(scores));mask=masks[best]
     array=run/'sam_arrays'/(stem+'.npz');tmp=array.with_suffix('.tmp')
     with tmp.open('wb') as f:np.savez_compressed(f,candidate_masks=masks,candidate_quality_scores=scores,low_res_logits=logits,
        box=np.asarray(prompt['box']),points=np.asarray(prompt['points']),point_labels=np.asarray(prompt['labels']),points_used=np.asarray(mode=='box_plus_points'),selected_index=np.asarray(best));f.flush();os.fsync(f.fileno())
     os.replace(tmp,array)
     candidates=[]
     for k,proposal in enumerate(masks):
      candidate_path=run/'masks'/(stem+f'_candidate{k}.png');Image.fromarray(proposal.astype(np.uint8)*255).save(candidate_path)
      candidates.append(dict(candidate=k,quality_score=float(scores[k]),pixels=int(proposal.sum()),roi_coverage=float((proposal*roi).sum()/roi.sum()),path=str(candidate_path.relative_to(run))))
     mask_path=run/'masks'/(stem+'_selected.png');Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
     overlay_path=run/'overlays'/(stem+'.png');Image.fromarray(overlay(rgb,mask)).save(overlay_path)
     area=int(mask.sum());pairs=[iou(masks[i],masks[j]) for i in range(3) for j in range(i+1,3)]
     r.update(status='complete',selected_candidate=best,sam_quality_score=float(scores[best]),mask_pixels=area,whole_image_fraction=area/(512*512),
       roi_coverage=float((mask*roi).sum()/roi.sum()),outside_roi_fraction=float((mask*(1-roi)).sum()/area) if area else None,
       lip_coverage=float((mask&lips).sum()/lips.sum()) if lips.sum() else None,candidate_mean_iou=float(np.mean(pairs)),candidate_min_iou=float(min(pairs)),
       mask_nonempty=bool(area),candidates=candidates,image_embedding_path=str(emb.relative_to(run)),image_embedding_sha256=sha(emb),
       array_path=str(array.relative_to(run)),array_sha256=sha(array),selected_mask_path=str(mask_path.relative_to(run)),selected_mask_sha256=sha(mask_path),
       overlay_path=str(overlay_path.relative_to(run)),overlay_sha256=sha(overlay_path),prompt_view_path=str(prompt_path.relative_to(run)))
    except BaseException:
     r.update(status='failed',error=traceback.format_exc());record(r);raise
    record(r);latest[(image['image_id'],mode)]=r
    print(image['image_id'],mode,'area',r['mask_pixels'],'quality',r['sam_quality_score'],flush=True)
   csv_write(run/'results/proposals.csv',[dict(v,candidates=json.dumps(v.get('candidates'))) for v in latest.values()])
  csv_write(run/'results/proposals.csv',[dict(v,candidates=json.dumps(v.get('candidates'))) for v in latest.values()])
  save_json(run/'results/inference_complete.json',dict(n_images=len(images[:a.limit] if a.limit else images),n_completed=sum(r['status']=='complete' for r in latest.values()),
    no_new_diffusion_images=True,no_weights_updated=True,mask_semantics_validated=False))
if __name__=='__main__':main()
