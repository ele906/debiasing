"""Replay discovery images and visualize detector-associated SAE features."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'backprop'))
from generate_images import BLOCK,CHECKPOINT
from heatmap_style import make_heatmap_overlay


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    out=HERE/'sae_comparison/maps'
    if out.exists():
        raise ValueError('Maps already exist; preserve earlier results')
    settings=json.loads((ROOT/'backprop/generation_settings.json').read_text())
    if hashlib.sha256((CHECKPOINT/'state_dict.pth').read_bytes()).hexdigest()!=settings['sae_weights_sha256']:
        raise ValueError('Checkpoint mismatch')
    rows={r['sample_id']:r for r in read(ROOT/'backprop/manifest.csv')}
    targets=read(HERE/'sae_comparison/detector_targets.csv')
    summary={r['concept']:r for r in read(HERE/'sae_comparison/summary.csv')}
    selected=[]
    for concept in ['headscarf','necklace','kimono']:
        for present in ['1','0']:
            candidates=[r for r in targets if r['concept']==concept and r['split']=='train' and r['detector_present']==present][:2]
            selected += [(concept,int(summary[concept]['feature']),present,rows[r['sample_id']]) for r in candidates]
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
    sae=SparseAutoencoder.load_from_disk(str(CHECKPOINT)).float().cuda().eval()
    out.mkdir(parents=True)
    metrics=[]
    for concept,feature,present,row in selected:
        seed=int(row['seed'])
        noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.float16,generator=torch.Generator('cuda').manual_seed(seed))
        with torch.no_grad():
            result,cache=pipe.run_with_cache(prompt=row['prompt'],latents=noise,generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=1,guidance_scale=0.,height=512,width=512,positions_to_cache=[BLOCK],save_input=True)
            residual=(cache['output'][BLOCK]-cache['input'][BLOCK])[:,-1].permute(0,2,3,1).float()
            values=sae.encode(residual)[0,...,feature].cpu().numpy()
        with Image.open(ROOT/'backprop'/row['image']) as src:
            original=src.convert('RGB')
        error=float(np.abs(np.asarray(original).astype(float)-np.asarray(result.images[0])).mean())
        if error!=0:
            raise RuntimeError(f'Replay mismatch {row["sample_id"]}: {error}')
        overlay=make_heatmap_overlay(original,values)
        sheet=Image.new('RGB',(768,420),'white')
        sheet.paste(original.resize((384,384)),(0,36))
        sheet.paste(overlay.resize((384,384)),(384,36))
        ImageDraw.Draw(sheet).text((5,8),f'{concept}, SAE {feature}, detector={present}, {row["sample_id"]}',fill='black')
        name=f'{concept}_{row["sample_id"]}'
        sheet.save(out/(name+'.jpg'),quality=90)
        np.save(out/(name+'.npy'),values)
        metrics.append(dict(concept=concept,feature=feature,detector_present=present,sample_id=row['sample_id'],mean=float(values.mean()),maximum=float(values.max()),active_fraction=float((values>0).mean()),replay_mae=error))
        print(f'{name}: mean={values.mean():.4g}, max={values.max():.4g}',flush=True)
        del cache,residual
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,lineterminator='\n',fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    (out/'README.md').write_text('Discovery images only; exact pixel replay verified. Left: original. Right: SAE map, scaled independently per image. Compare metrics.csv for magnitudes; brightness across maps is not comparable. Detector positives/negatives are unreviewed. These maps do not establish a causal concept feature.\n')


if __name__=='__main__':
    main()
