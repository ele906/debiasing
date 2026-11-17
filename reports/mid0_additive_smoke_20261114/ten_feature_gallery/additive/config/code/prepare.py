"""Freeze existing feature4228 additive/multiplicative scoring inputs; no generation."""
import csv,datetime as dt,hashlib,json,os,shutil,subprocess
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path('/n/fs/goose');BASE=ROOT/'experiments/mid0_sae_selectivity';SAE='cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def write(p,rs):
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def main():
 mult=ROOT/'layers/mid0_stability/combined20_results/per_image.csv';add=BASE/'ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv';addcfg=BASE/'additive_20261114_174738_954258/config/experiment_config.json'
 acfg=json.loads(addcfg.read_text());assert acfg['sae_sha256']==SAE
 cfgpaths=[ROOT/'layers/runs/mid0_sae_face_001/interventions'/d/'feature_4228/settings.json' for d in ['stability_top10_4243081','stability_extra15_4243220']]
 for p in cfgpaths:
  c=json.loads(p.read_text());assert c['sae_weights_sha256']==SAE and c['block']=='unet.mid_block.attentions.0'
 run=BASE/('beard_interventions_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 for d in ['config/code','config/source','originals','assets','logs','results','figures','smoke']:(run/d).mkdir(parents=True,exist_ok=True)
 records=[];mrs=[r for r in read(mult) if r['feature']=='4228'];ars=[r for r in read(add) if r['feature_id']=='4228'];assert len(mrs)==6960 and len(ars)==420
 for r in mrs:
  alpha=float(r['alpha']);records.append(dict(condition_id=f"mult_{r['group']}_{r['seed']}_{alpha:g}",method='multiplicative',prompt_group=r['group'],seed=int(r['seed']),strength_label=f'alpha_{alpha:g}',multiplier=alpha,additive_offset='',source_path=r['image'],source_sha256=r['image_sha256'],original_activation_nonzero_fraction=r['active_fraction'],original_activation_zero=not (r['initially_active']=='True'),applied_l2=r['delta_fro_applied'],face_detected=r['face_detected'],source_evaluation=str(mult),activation_path='',negative_multiplier=alpha<0))
 for r in ars:
  records.append(dict(condition_id='add_'+r['image_id'],method='additive',prompt_group=r['group'],seed=int(r['seed']),strength_label=r['strength_label'],multiplier='',additive_offset=float(r['additive_offset']),source_path=r['image_path'],source_sha256=r['image_sha256'],original_activation_nonzero_fraction=r['original_activation_nonzero_fraction'],original_activation_zero=r['original_activation_zero'],applied_l2=r['perturbation_norm'],face_detected='',source_evaluation=str(add),activation_path=r['activation_path'],source_run=r['run_id'],negative_multiplier=False))
 for group,seed in sorted({(r['group'],int(r['seed'])) for r in ars}):
  r=next(r for r in ars if r['group']==group and int(r['seed'])==seed)
  records.append(dict(condition_id=f'baseline_{group}_{seed}',method='baseline',prompt_group=group,seed=seed,strength_label='baseline',multiplier='',additive_offset=0,source_path=r['baseline_image_path'],source_sha256=sha(r['baseline_image_path']),applied_l2=0,negative_multiplier=False))
 assert len(records)==7440
 unique={};index={(r['prompt_group'],r['seed']):r for r in records if r['method']=='multiplicative' and r['multiplier']==1}
 for i,r in enumerate(records):
  p=Path(r['source_path']);h=r['source_sha256'];assert sha(p)==h,(r['condition_id'],'source changed')
  if h not in unique:
   with Image.open(p) as im:assert im.size==(512,512)
   dest=run/'originals'/(h+'.png');os.link(p,dest);unique[h]=dict(asset_id=h,image_path=str(dest.relative_to(run)),source_path=str(p),source_sha256=h)
  r['asset_id']=h;r['image_path']=unique[h]['image_path'];r['feature_id']=4228;r['layer']='unet.mid_block.attentions.0';r['sae_sha256']=SAE
  if r['method']=='multiplicative':r['control_condition_id']=index[(r['prompt_group'],r['seed'])]['condition_id']
  elif r['method']=='additive':r['control_condition_id']=f"baseline_{r['prompt_group']}_{r['seed']}"
  else:r['control_condition_id']=r['condition_id']
  if i%1000==0:print('Verified sources',i,'/',len(records),flush=True)
 # Compare additive no-intervention images to multiplicative alpha1 by actual RGB pixels.
 controls=[]
 for r in records:
  if r['method']!='baseline':continue
  m=index[r['prompt_group'],r['seed']]
  with Image.open(r['source_path']) as a,Image.open(m['source_path']) as b:equal=np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
  controls.append(dict(prompt_group=r['prompt_group'],seed=r['seed'],additive_baseline=r['condition_id'],multiplicative_control=m['condition_id'],pixels_equal=equal))
 write(run/'config/control_equivalence.csv',controls)
 write(run/'config/conditions.csv',records);write(run/'config/assets.csv',list(unique.values()))
 # Reuse exact-input full-resolution hard-class counts and binary masks, not missing probabilities.
 olddirs=[BASE/'faceparser_beard_20seed',BASE/'faceparser_beard_other_groups'];reuse=[]
 for old in olddirs:
  for r in read(old/'beard_class_metrics.csv'):
   h=r['source_sha256']
   if h not in unique:continue
   stem=f"{r['prompt_group']}__seed{int(r['seed']):03d}__alpha_{r['alpha_label']}";mask=old/'masks'/(stem+'_class14_beard.png');assert mask.exists()
   with Image.open(mask) as im:arr=np.asarray(im)>0;assert arr.shape==(512,512) and int(arr.sum())==int(r['beard_pixels'])
   dest=run/'assets'/(h+'_beard.png')
   if dest.exists():continue
   os.link(mask,dest)
   result=dict(asset_id=h,status='complete',prediction_origin='reused_exact_input',reused_from=str(old),reused_metrics_sha256=sha(old/'beard_class_metrics.csv'),beard_mask_path=str(dest.relative_to(run)),beard_mask_sha256=sha(dest),
    beard_pixels=int(r['beard_pixels']),image_pixels=262144,beard_fraction_image=float(r['beard_fraction_image']),face_classes_1_to_14_pixels=int(r['face_classes_1_to_14_pixels']),
    beard_fraction_face_classes_1_to_14=float(r['beard_fraction_face_classes_1_to_14']),probability_available=False,probabilities_path='',full_class_map_available=False,
    **{f'class_{i}_pixels':int(r[f'class_{i}_pixels']) for i in range(19)})
   (run/'assets'/(h+'.json')).write_text(json.dumps(result,indent=2));reuse.append(dict(asset_id=h,source=str(mask),sha256=sha(mask)))
 write(run/'config/reuse_manifest.csv',reuse)
 repo=ROOT/'face-parsing-beard-class';weights=[repo/'experiments/checkpoints/decoder.ckpt',repo/'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'];hashes=[sha(p) for p in weights]
 assert hashes==['ad5266f61eaf540f120e15c47315bef3e2302c720c1c7135d0a3e9f9de55af39','73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c']
 helper=ROOT/'instance-segmentation/run_beard_class_faceparser.py';shutil.copy2(helper,run/'config/code/parser_helpers.py')
 for p in Path(__file__).parent.iterdir():
  if p.suffix in ['.py','.sbatch']:shutil.copy2(p,run/'config/code'/p.name)
 cfg=dict(run_id=run.name,feature_id=4228,layer='unet.mid_block.attentions.0',sae_sha256=SAE,parser_repo=str(repo),parser_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    checkpoint_paths=[str(p) for p in weights],checkpoint_hashes=hashes,class_id=14,class_mapping='README lists Beard; visualization helper conflicts. Raw class14 is retained unchanged. Not ground-truth masks.',
    n_conditions=len(records),n_unique_assets=len(unique),n_reused_assets=len(reuse),n_shards=4,seeds_multiplicative=list(range(30,50)),seeds_additive=list(range(30,35)),
    expected_storage_upper_gb=len(unique)*(512*512*5+300000)/1e9,probability_storage='New predictions: float32 class14 probabilities, uint8 full argmax class map, class mean probabilities. Reused results: binary masks and class counts only; unavailable probabilities remain missing.',
    evaluation='Identical existing helper: RGB 512x512, PIL bilinear, ImageNet normalization, float32, argmax probabilities, no threshold tuning.',n_generated_images=0,n_models_trained=0)
 (run/'config/experiment_config.json').write_text(json.dumps(cfg,indent=2))
 for i,p in enumerate([mult,add,addcfg,*cfgpaths]):shutil.copy2(p,run/'config/source'/(str(i)+'_'+p.name))
 inputs=[mult,add,addcfg,*cfgpaths,*weights,helper,repo/'README.md',repo/'src/utils/visualization.py',repo/'src/models/decoder.py',repo/'src/models/encoder.py']
 write(run/'config/input_hashes.csv',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs])
 (run/'config/git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True));(run/'config/git_status.txt').write_text(subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True))
 print('RUN_DIRECTORY',run,flush=True);print(json.dumps(cfg,indent=2),flush=True)
if __name__=='__main__':main()
