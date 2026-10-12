"""Generate fresh paired controls for three fixed candidate SAE features."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'independent_controls'
FEATURES={'necklace':3181,'afro':1500,'beard':2893}
GROUPS={'necklace':[('indian','man'),('hispanic','woman')],'afro':[('black','woman'),('white','woman')],'beard':[('east asian','man'),('hispanic','man')]}
DESCRIPTIONS={'necklace':('wearing a plain crew-neck shirt and a gold chain necklace','wearing a plain crew-neck shirt without any jewelry'),'afro':('with an afro hairstyle, wearing a plain shirt','with long straight hair, wearing a plain shirt'),'beard':('with a full beard, wearing a plain shirt','with a clean-shaven face, wearing a plain shirt')}


def main():
    if OUT.exists():raise ValueError('Preserve existing control generation')
    settings=json.loads((ROOT/'backprop/generation_settings.json').read_text())
    checkpoint=Path(settings['sae_checkpoint'])
    if hashlib.sha256((checkpoint/'state_dict.pth').read_bytes()).hexdigest()!=settings['sae_weights_sha256']:raise ValueError('SAE hash mismatch')
    rows=[]
    for concept,groups in GROUPS.items():
        for race,gender in groups:
            for seed in range(1000,1004):
                for condition,description in zip(['requested_present','requested_absent'],DESCRIPTIONS[concept]):
                    sample=f'{concept}_{race.replace(" ","_")}_{gender}_{condition}_seed{seed}'
                    rows.append(dict(sample_id=sample,race_prompt=race,gender_prompt=gender,seed=seed,split='independent',concept=concept,condition=condition,prompt=f'a head-and-shoulders studio photo of a {race} {gender} {description}',image=str(OUT/'images'/(sample+'.png')),features=str(OUT/'features'/(sample+'.npz')),visual_presence='',reviewer='',notes=''))
    OUT.mkdir();(OUT/'images').mkdir();(OUT/'features').mkdir()
    with (OUT/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (OUT/'settings.json').write_text(json.dumps(dict(generation=settings,features=FEATURES,groups=GROUPS,descriptions=DESCRIPTIONS,seeds=[1000,1001,1002,1003],samples=len(rows),purpose='Fresh paired controls; requested condition is not a ground-truth label. Images must be visually reviewed. Features are fixed exploratory choices from earlier data.'),indent=2)+'\n')
    sys.path.insert(0,str(ROOT/'sdxl-unbox'))
    import torch
    from PIL import Image,ImageDraw
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    torch.set_num_threads(4)
    load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
    vae=AutoencoderKL.from_pretrained(settings['vae'],**load)
    pipe=HookedStableDiffusionXLPipeline.from_pretrained(settings['model'],vae=vae,variant='fp16',**load)
    pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing')
    pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
    sae=SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval();sae.requires_grad_(False)
    block=settings['block'];metrics=[]
    for i,row in enumerate(rows):
        seed=row['seed'];feature=FEATURES[row['concept']]
        noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.float16,generator=torch.Generator('cuda').manual_seed(seed))
        with torch.inference_mode():
            result,cache=pipe.run_with_cache(prompt=row['prompt'],latents=noise,generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=1,guidance_scale=0.,height=512,width=512,positions_to_cache=[block],save_input=True)
            residual=(cache['output'][block]-cache['input'][block])[:,-1].permute(0,2,3,1).float()
            z=sae.encode(residual)[0]
            selected=z[...,feature].cpu().numpy()
            np.savez_compressed(row['features'],mean=z.mean((0,1)).cpu().numpy(),maximum=z.amax((0,1)).cpu().numpy(),active_fraction=(z>0).float().mean((0,1)).cpu().numpy(),map=selected,feature=np.array(feature))
        result.images[0].save(row['image'])
        metrics.append(dict(sample_id=row['sample_id'],concept=row['concept'],condition=row['condition'],feature=feature,mean=float(selected.mean()),maximum=float(selected.max())))
        del cache,residual,z
        print(f'{i+1}/{len(rows)} fresh paired images generated',flush=True)
    with (OUT/'sae_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');w.writeheader();w.writerows(metrics)
    for concept in FEATURES:
        samples=[r for r in rows if r['concept']==concept]
        sheet=Image.new('RGB',(4*256,4*294),'white');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(samples):
            x,y=(i%4)*256,(i//4)*294
            with Image.open(row['image']) as image:sheet.paste(image.resize((256,256)),(x,y+38))
            draw.text((x+3,y+3),f'{row["race_prompt"]} {row["gender_prompt"]} seed {row["seed"]}',fill='black')
            draw.text((x+3,y+18),row['condition'],fill='black')
        sheet.save(OUT/(concept+'_controls.jpg'),quality=90)
    print('Fresh paired control images ready for visual review',flush=True)


if __name__=='__main__':main()
