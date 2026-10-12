"""Sweep a fixed SAE feature coefficient with an activity-matched control."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def mask_for(record,pattern):
    mask=np.zeros((512,512),dtype=bool)
    for source in ['grounding','objects']:
        for label,box in zip(record[source]['labels'],record[source]['bboxes']):
            if not re.search(pattern,str(label).lower()) or len(box)!=4 or not np.isfinite(box).all():continue
            x1,y1,x2,y2=box
            if x2<x1 or y2<y1:continue
            x1,y1=min(512,max(0,int(np.floor(x1)))),min(512,max(0,int(np.floor(y1))))
            x2,y2=min(512,max(0,int(np.ceil(x2)))),min(512,max(0,int(np.ceil(y2))))
            mask[y1:y2,x1:x2]=True
    return mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'run_001')
    p.add_argument('--feature',type=int,default=3181)
    p.add_argument('--strengths',type=float,nargs='+',default=[0,.5,1,1.5])
    args=p.parse_args()
    if not all(np.isfinite(args.strengths)) or min(args.strengths)<0 or 1 not in args.strengths:p.error('Finite nonnegative strengths including 1 required')
    if len(set(args.strengths))!=len(args.strengths):p.error('Duplicate strengths')
    if args.output.exists():raise ValueError('Use a new output to preserve earlier sweeps')
    data=ROOT/'concept/independent_controls'
    rows=[r for r in read(data/'manifest.csv') if r['concept']=='necklace']
    if len(rows)!=16:raise ValueError('Expected 16 fresh necklace-control images')
    source_settings=json.loads((data/'settings.json').read_text())
    settings=source_settings['generation']
    checkpoint=Path(settings['sae_checkpoint'])
    if hashlib.sha256((checkpoint/'state_dict.pth').read_bytes()).hexdigest()!=settings['sae_weights_sha256']:raise ValueError('Checkpoint changed')
    pooled=[]
    for row in rows:
        with np.load(row['features'],allow_pickle=False) as f:pooled.append(f['mean'])
    pooled=np.stack(pooled)
    if not 0<=args.feature<pooled.shape[1]:raise ValueError('Feature out of range')
    activity=(pooled>0).mean(0);means=pooled.mean(0)
    exclude={int(r['feature']) for r in read(ROOT/'concept/sae_comparison_full/candidates.csv')}
    eligible=[j for j in range(pooled.shape[1]) if j not in exclude and abs(activity[j]-activity[args.feature])<=.25 and .5*means[args.feature]<=means[j]<=2*means[args.feature] and means[j]>0]
    if not eligible:raise ValueError('No activity-matched control found; define another control explicitly')
    control=int(np.random.default_rng(20261012).choice(eligible))
    ids=[args.feature,control]
    args.output.mkdir(parents=True);(args.output/'images').mkdir();(args.output/'grids').mkdir()
    run_settings=dict(source=str(data),generation=settings,target_feature=args.feature,random_control=control,control_candidates=len(eligible),control_selection='Seeded random draw excluding earlier concept candidates; mean activation within factor 2 and active-image fraction within 0.25 of target on these 16 baselines',control_rng_seed=20261012,strengths=args.strengths,samples=len(rows),edit='h + (alpha - 1) * z_j * decoder_direction_j',interpretation='Exploratory fixed interventions; no model or SAE training; requested conditions and detector labels need visual review')
    (args.output/'settings.json').write_text(json.dumps(run_settings,indent=2)+'\n')
    sys.path.insert(0,str(ROOT/'sdxl-unbox'));sys.path.insert(0,str(ROOT/'sdxl-unbox/scripts'))
    import torch
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    from sae_experiment import intervention
    torch.set_num_threads(4)
    load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
    vae=AutoencoderKL.from_pretrained(settings['vae'],**load)
    pipe=HookedStableDiffusionXLPipeline.from_pretrained(settings['model'],vae=vae,variant='fp16',**load)
    pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing')
    pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
    sae=SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval();sae.requires_grad_(False)
    metrics,review,audit=[],[],[]
    def kwargs(row):
        seed=int(row['seed'])
        noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.float16,generator=torch.Generator('cuda').manual_seed(seed))
        return dict(prompt=row['prompt'],latents=noise,generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=1,guidance_scale=0.,height=512,width=512)
    def audit_row(row,sample,path):
        return dict(sample_id=sample,split='independent',race_prompt=row['race_prompt'],gender_prompt=row['gender_prompt'],seed=row['seed'],prompt=row['prompt'],image=str(path.resolve()),features='')
    for index,row in enumerate(rows):
        sample=row['sample_id'];base_id=sample+'_baseline';base_path=args.output/'images'/(base_id+'.png')
        with Image.open(row['image']) as src:saved=src.convert('RGB')
        baseline=pipe.pipe(**kwargs(row)).images[0]
        saved_error=float(np.abs(np.asarray(saved).astype(float)-np.asarray(baseline)).mean())
        if saved_error!=0:raise RuntimeError(f'Original replay mismatch: {sample} MAE {saved_error}')
        baseline.save(base_path);audit.append(audit_row(row,base_id,base_path))
        record=json.loads((data/'detector'/(sample+'.json')).read_text())
        if record['image_sha256']!=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest():raise ValueError('Detector image changed')
        concept_mask=mask_for(record,r'\bnecklace\b');face_mask=mask_for(record,r'\bface\b')
        sheets=[]
        for feature in ids:
            role='target' if feature==args.feature else 'control'
            tiles=[]
            for strength in args.strengths:
                edited=pipe.run_with_hooks(**kwargs(row),position_hook_dict={settings['block']:intervention(sae,[feature],strength)}).images[0]
                delta=np.abs(np.asarray(edited).astype(float)-np.asarray(baseline)).mean(2)
                change=float(delta.mean())
                if strength==1 and change!=0:raise RuntimeError('Strength 1 failed exact no-op identity')
                if strength==1:
                    edited_id=base_id;path=base_path
                else:
                    edited_id=sample+f'_{role}_feature{feature}_alpha{strength:g}'.replace('.','p')
                    path=args.output/'images'/(edited_id+'.png');edited.save(path);audit.append(audit_row(row,edited_id,path))
                result=dict(sample_id=sample,edited_sample_id=edited_id,race_prompt=row['race_prompt'],gender_prompt=row['gender_prompt'],condition=row['condition'],seed=row['seed'],feature=feature,role=role,strength=strength,baseline_feature_mean=float(pooled[index,feature]),pixel_mae=change,necklace_box_mae='' if not concept_mask.any() else float(delta[concept_mask].mean()),outside_necklace_box_mae='' if not concept_mask.any() or concept_mask.all() else float(delta[~concept_mask].mean()),face_box_mae='' if not face_mask.any() else float(delta[face_mask].mean()),saved_baseline_replay_mae=saved_error)
                metrics.append(result)
                review.append(dict(**result,original=str(base_path.resolve()),edited=str(path.resolve()),necklace_presence='',necklace_change='',face_damage='',clothing_damage='',artifacts='',prompt_fidelity='',reviewer='',notes=''))
                tiles.append((f'{role} {feature}: alpha {strength:g}',edited))
            sheet=Image.new('RGB',(len(tiles)*256,288),'white');draw=ImageDraw.Draw(sheet)
            for i,(label,image) in enumerate(tiles):
                sheet.paste(image.resize((256,256)),(i*256,32));draw.text((i*256+3,8),label,fill='black')
            sheets.append(sheet)
        combined=Image.new('RGB',(len(args.strengths)*256,608),'white');draw=ImageDraw.Draw(combined)
        draw.text((3,6),sample,fill='black')
        for i,sheet in enumerate(sheets):combined.paste(sheet,(0,32+i*288))
        combined.save(args.output/'grids'/(sample+'.jpg'),quality=90)
        print(f'{index+1}/{len(rows)} feature sweeps complete',flush=True)
    write(args.output/'metrics.csv',metrics);write(args.output/'review.csv',review);write(args.output/'audit_manifest.csv',audit)
    report=['# Feature-strength tuning pilot','','Sixteen fresh necklace-control images, fixed target feature and activity-matched random control. Both model and SAE weights stay frozen.','',f'Target: {args.feature}; control: {control}; strengths: {args.strengths}.','', 'Alpha 1 is unchanged, 0.5 retains half the feature contribution, 0 removes it, and 1.5 amplifies it. The edit preserves the SAE reconstruction residual. All sixteen original replays and every alpha-1 hook output must match exactly.','', 'Pixel differences measure edits, not concept removal or reduced racism. Face and clothing changes must be reviewed. The detector audit is a proxy with known label errors; choose strengths after visual review rather than declaring the lowest detector score optimal.','', '## Comparison grids','']
    for row in rows:
        report += [f'### {row["race_prompt"]} {row["gender_prompt"]}, {row["condition"]}, seed {row["seed"]}', '',f'![strength sweep](grids/{row["sample_id"]}.jpg)','']
    (args.output/'REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    print('Sweep complete; detector audit and visual review remain',flush=True)


if __name__=='__main__':main()
