"""Prepare 12 groups x 3 existing conditions; frozen baseline prompts, no generation."""
import argparse,datetime as dt,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,SAE_HASH,sha,save_json,csv_write,prompt_geometry

SOURCE=ROOT/'experiments/mid0_sae_selectivity/additive_20261114_174738_954258'
METRICS=ROOT/'experiments/mid0_sae_selectivity/ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv'
SAM=ROOT/'segment-anything/checkpoints/sam_vit_b_01ec64.pth'

def main():
 import csv
 cfg=json.loads((SOURCE/'config/experiment_config.json').read_text());assert cfg['sae_sha256']==SAE_HASH
 with METRICS.open() as f:rows=[r for r in csv.DictReader(f) if r['feature_id']=='4228' and r['group'] in ('indian_man','middle_eastern_man') and r['seed'] in ('30','31','32','33','34') and r['strength_label'] in ('fixed_10','fixed_50')]
 assert len(rows)==20
 for r in rows:r['prompt_group']=r['group'];r['group']=r['group']+'_seed'+r['seed']
 groups=sorted({r['group'] for r in rows});assert len(groups)==10
 run=ROOT/'experiments/mid0_sae_selectivity'/('sam_beard_two_groups_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 for d in ('config/code','originals','landmarks','masks','overlays','prompt_views','sam_arrays','image_embeddings','results','figures','logs'):(run/d).mkdir(parents=True,exist_ok=True)
 manifest=[]
 for group in groups:
  subset=[r for r in rows if r['group']==group]
  for condition in ('baseline','fixed_10','fixed_50'):
   r=subset[0] if condition=='baseline' else next(r for r in subset if r['strength_label']==condition)
   source=Path(r['baseline_image_path'] if condition=='baseline' else r['image_path'])
   if condition!='baseline':assert sha(source)==r['image_sha256']
   name=f'{group}_{condition}';dest=run/'originals'/(name+'.png');os.link(source,dest)
   with Image.open(dest) as im:assert im.size==(512,512)
   manifest.append(dict(image_id=name,group=group,prompt_group=r['prompt_group'],seed=int(r['seed']),condition=condition,offset=0 if condition=='baseline' else int(condition.split('_')[1]),
      source_path=str(source),image_path=str(dest.relative_to(run)),image_sha256=sha(dest),source_image_id='' if condition=='baseline' else r['image_id']))
 csv_write(run/'config/images.csv',manifest)
 options=dict(run_id=run.name,feature_id=4228,layer='unet.mid_block.attentions.0',sae_checkpoint=cfg['sae_checkpoint'],sae_sha256=SAE_HASH,
    generator='Existing frozen SDXL Turbo images only; no generator/SAE inference',sam_checkpoint=str(SAM),sam_sha256=sha(SAM),sam_architecture='vit_b',
    groups=groups,seeds=list(range(30,35)),conditions=['baseline','fixed_10','fixed_50'],prompt_sets=['box_only','box_plus_points'],selection='Exploratory Indian-man and Middle-Eastern-man subset selected after inspecting seed30 pilot; all five seeds retained',
    inference_rule='SAM model-quality argmax per prompt, preserve all3 candidates and logits',presence_gate='None: nonempty masks are proposals, not evidence of beard presence',
    definitions='Candidate chin/jaw region excludes lips/moustache by negative prompting intent; semantic membership unverified',
    metrics='Raw proposal area, overlap with fixed lower-face ROI and lips, prompt sensitivity, paired proposal-area changes; NOT validated beard coverage')
 save_json(run/'config/experiment_config.json',options)
 # Existing geometric ROI, loaded without its optional MediaPipe deps shadowing NumPy/Matplotlib.
 file=ROOT/'face-landmarker/render_landmarked_lower_face_change_report.py'
 spec=importlib.util.spec_from_file_location('existing_face_roi',file);roi=importlib.util.module_from_spec(spec);sys.modules[spec.name]=roi;spec.loader.exec_module(roi)
 mp,detector=roi.build_detector(ROOT/'face-landmarker/models/face_landmarker.task',ROOT/'face-landmarker/.deps')
 detections={};geometry={};statuses=[]
 with detector:
  for r in sorted(manifest,key=lambda r:(r['condition']!='baseline',r['image_id'])):
   image=run/r['image_path'];result=detector.detect(mp.Image.create_from_file(str(image)))
   status=dict(image_id=r['image_id'],detected=bool(result.face_landmarks),mediapipe_version=mp.__version__)
   if not result.face_landmarks:
    status['status']='no_landmarks';statuses.append(status);continue
   points=np.asarray([[p.x,p.y,p.z] for p in result.face_landmarks[0]],np.float32);detections[r['image_id']]=points
   np.savez_compressed(run/'landmarks'/(r['image_id']+'.npz'),landmarks=points)
   if r['condition']=='baseline':
    conf=prompt_geometry(points,512,512);full,polygon=roi.beard_roi_from_landmarks(result.face_landmarks[0],512,512)
    lips=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
    canvas=Image.new('L',(512,512),0);ImageDraw.Draw(canvas).polygon([(float(points[i,0]*512),float(points[i,1]*512)) for i in lips],fill=255)
    np.savez_compressed(run/'landmarks'/(r['group']+'_regions.npz'),roi=full,lips=np.asarray(canvas)>0,polygon=np.asarray(polygon))
    geometry[r['group']]=conf
   base=detections.get(r['group']+'_baseline')
   status.update(status='complete',baseline_landmark_mean_drift_px=0. if r['condition']=='baseline' else None if base is None else float(np.linalg.norm((points[:,:2]-base[:,:2])*512,axis=1).mean()))
   statuses.append(status)
 save_json(run/'config/frozen_prompts.json',geometry);csv_write(run/'results/landmark_status.csv',statuses)
 save_json(run/'config/landmark_model.json',dict(checkpoint=str(ROOT/'face-landmarker/models/face_landmarker.task'),sha256=sha(ROOT/'face-landmarker/models/face_landmarker.task'),mediapipe=mp.__version__,numpy=np.__version__,
    frozen_prompts='Generated from baseline only and reused unchanged at both offsets; groups/genders do not alter the geometry rule'))
 inputs=[METRICS,SOURCE/'config/experiment_config.json',SAM,file,ROOT/'instance-segmentation/evaluate_sam_beard_prompts.py']
 for p in Path(__file__).parent.glob('*.py'):
  target=run/'config/code'/p.name;shutil.copy2(p,target);inputs.append(target)
 csv_write(run/'config/input_hashes.csv',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs])
 (run/'config/preparation_environment.txt').write_text(sys.version+'\n'+subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
 print('RUN_DIRECTORY',run,flush=True);print('Images',len(manifest),'groups',len(groups),'baseline prompts',len(geometry),flush=True)
if __name__=='__main__':main()
