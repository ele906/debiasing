"""A small, fixed-seed intervention pilot for turban, flower, and kimono candidates."""
import argparse, csv, json, re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'wardrobe_pilot'
TARGETS={'turban':(1526,'middle eastern','man',r'\bturban\b',4,4),
         'flower':(2230,'east asian','woman',r'\bflowers?\b',4,4),
         'kimono':(5046,'east asian','man',r'\bkimono\b',2,2)}

def rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))
def write(path, rs):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--render',action='store_true');a=p.parse_args()
    if a.prepare:
        preds=rows(ROOT/'concept/concept_predictions.csv')
        selected=[]
        for name,(feature,race,gender,pattern,npos,nneg) in TARGETS.items():
            group=[r for r in preds if r['split']=='validation' and r['race_prompt']==race and r['gender_prompt']==gender]
            pos=[r for r in group if re.search(pattern,r['detected_concepts'].lower())]
            neg=[r for r in group if not re.search(pattern,r['detected_concepts'].lower())]
            if len(pos)<npos or len(neg)<nneg:raise ValueError(f'Not enough same-group examples for {name}: {len(pos)}+, {len(neg)}-')
            rng=np.random.default_rng(20261012+feature)
            chosen=[(r,'detector_positive') for r in rng.choice(pos,npos,replace=False)]+[(r,'detector_negative') for r in rng.choice(neg,nneg,replace=False)]
            for r,label in chosen:
                selected.append(dict(concept=name,target_feature=feature,source_label=label,**{k:r[k] for k in ['sample_id','split','race_prompt','gender_prompt','image','features']}))
        OUT.mkdir(exist_ok=True)
        write(OUT/'manifest.csv',selected)
        (OUT/'settings.json').write_text(json.dumps({'targets':{k:{'feature':v[0],'group':[v[1],v[2]],'pattern':v[3],'positive_n':v[4],'negative_n':v[5]} for k,v in TARGETS.items()},'selection':'Same-prompt validation examples, fixed seeded sample; labels are unreviewed Florence-2 detector outputs','strengths':[0,0.5,1]},indent=2)+'\n')
        print('Prepared',len(selected),'same-prompt validation examples')
    if a.run:
        run()
    if a.render:
        render()

def render():
    from PIL import Image,ImageDraw
    manifest=rows(OUT/'manifest.csv');settings=json.loads((OUT/'settings.json').read_text())
    targets=settings['targets'];controls=settings['random_controls'];strengths=settings['strengths']
    for r in manifest:
        sheet=Image.new('RGB',(len(strengths)*256,18+2*294),'white');draw=ImageDraw.Draw(sheet)
        draw.text((3,3),f'{r["concept"]}: {r["sample_id"]} ({r["source_label"]})',fill='black')
        for row,(role,feature) in enumerate([('target',targets[r['concept']]),('random_control',controls[r['concept']])]):
            for col,alpha in enumerate(strengths):
                label=f'{role} f{feature} alpha {alpha:g}';x,y=col*256,row*294+18
                draw.text((x+3,y+3),label,fill='black')
                image=Image.open(OUT/'images'/f'{r["sample_id"]}_{role}_f{feature}_a{alpha:g}.png').convert('RGB')
                sheet.paste(image.resize((256,256)),(x,y+20))
        sheet.save(OUT/'grids'/f'{r["sample_id"]}.jpg',quality=90)

def run():
    import hashlib,sys
    import torch
    from PIL import Image,ImageDraw
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    sys.path.insert(0,str(ROOT/'sdxl-unbox'));sys.path.insert(0,str(ROOT/'sdxl-unbox/scripts'))
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    from sae_experiment import intervention
    manifest=rows(OUT/'manifest.csv');settings=json.loads((ROOT/'backprop/generation_settings.json').read_text())
    ck=Path(settings['sae_checkpoint'])
    if hashlib.sha256((ck/'state_dict.pth').read_bytes()).hexdigest()!=settings['sae_weights_sha256']:raise ValueError('SAE checkpoint hash changed')
    pooled=np.load(ROOT/'backprop/sae_features.npz',allow_pickle=False);ids=list(pooled['sample_id'].astype(str));means=pooled['X_mean'];index={s:i for i,s in enumerate(ids)}
    targets={k:v[0] for k,v in TARGETS.items()};controls={}
    rng=np.random.default_rng(20261012)
    for name,feature in targets.items():
        samples=[r for r in manifest if r['concept']==name];ix=[index[r['sample_id']] for r in samples];x=means[ix];act=(x>0).mean(0);mu=x.mean(0)
        chosen=set(targets.values());eligible=[j for j in range(x.shape[1]) if j not in chosen and abs(act[j]-act[feature])<=.25 and mu[j]>0 and .5*mu[feature]<=mu[j]<=2*mu[feature]]
        if not eligible:raise ValueError(f'No activity-matched random feature for {name}')
        controls[name]=int(rng.choice(eligible))
    imdir=OUT/'images';grdir=OUT/'grids';imdir.mkdir(exist_ok=True);grdir.mkdir(exist_ok=True)
    torch.set_num_threads(4);load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
    vae=AutoencoderKL.from_pretrained(settings['vae'],**load);pipe=HookedStableDiffusionXLPipeline.from_pretrained(settings['model'],vae=vae,variant='fp16',**load)
    pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing');pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
    sae=SparseAutoencoder.load_from_disk(str(ck)).float().cuda().eval();sae.requires_grad_(False)
    report=['# Turban, flower, and kimono feature intervention pilot','','Same-prompt validation images replayed from their original prompts and seeds. Candidate SAE feature and matched random control are attenuated at alpha 0, 0.5, and 1. Alpha 1 is unchanged. Source detector labels are unreviewed; compare images directly.','',f'Random controls selected by pooled activity on this small set: {controls}.','']
    metric=[]
    for i,r in enumerate(manifest):
        seed=int(Path(r['image']).stem[-3:]);kwargs=dict(prompt=f'a picture of a {r["race_prompt"]} {r["gender_prompt"]}',latents=torch.randn((1,4,64,64),device='cuda',dtype=torch.float16,generator=torch.Generator('cuda').manual_seed(seed)),generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=1,guidance_scale=0.,height=512,width=512)
        base=pipe.pipe(**kwargs).images[0];source=Image.open(r['image']).convert('RGB');err=float(np.abs(np.asarray(base).astype(float)-np.asarray(source).astype(float)).mean())
        if err:raise RuntimeError(f'Exact source replay mismatch: {r["sample_id"]} MAE={err}')
        sample=[]
        for role,feature in [('target',targets[r['concept']]),('random_control',controls[r['concept']])]:
            tiles=[]
            for alpha in (0,.5,1):
                image=pipe.run_with_hooks(**kwargs,position_hook_dict={settings['block']:intervention(sae,[feature],alpha)}).images[0]
                delta=float(np.abs(np.asarray(image).astype(float)-np.asarray(base).astype(float)).mean())
                if alpha==1 and delta!=0:raise RuntimeError('Alpha 1 failed exact no-op')
                path=imdir/f'{r["sample_id"]}_{role}_f{feature}_a{alpha:g}.png';image.save(path)
                tiles.append((f'{role} f{feature} alpha {alpha:g}',image))
                metric.append(dict(concept=r['concept'],sample_id=r['sample_id'],source_label=r['source_label'],feature=feature,role=role,alpha=alpha,pixel_mae=delta,source_replay_mae=err,image=str(path.resolve())))
            sample.append(tiles)
        sheet=Image.new('RGB',(3*256,18+2*294),'white');draw=ImageDraw.Draw(sheet);draw.text((3,3),f'{r["concept"]}: {r["sample_id"]} ({r["source_label"]})',fill='black')
        for row,tiles in enumerate(sample):
            for col,(label,image) in enumerate(tiles):
                x,y=col*256,row*294+18;sheet.paste(image.resize((256,256)),(x,y+20));draw.text((x+3,y+3),label,fill='black')
        sheet.save(grdir/f'{r["sample_id"]}.jpg',quality=90)
        print(f'{i+1}/{len(manifest)} {r["sample_id"]}',flush=True)
    write(OUT/'metrics.csv',metric)
    for name in targets:
        report += [f'## {name}: feature {targets[name]}, random control {controls[name]}','']
        for r in manifest:
            if r['concept']==name:report += [f'### {r["sample_id"]} ({r["source_label"]})','','![Intervention comparison](grids/'+r['sample_id']+'.jpg)','']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    (OUT/'settings.json').write_text(json.dumps({'source':'Original validation images, exact prompt/seed replay','targets':targets,'random_controls':controls,'strengths':[0,.5,1],'samples':len(manifest),'interpretation':'Exploratory feature ablation; no model training. Detector labels are unreviewed and pixel change is not concept removal.'},indent=2)+'\n')

if __name__=='__main__':main()
