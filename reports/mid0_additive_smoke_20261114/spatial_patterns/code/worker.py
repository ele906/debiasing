"""One seed worker, paired contexts; immutable condition IDs and lossless per-attempt archives."""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from PIL import Image
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity')
sys.path.insert(0,str(BASE))
from run import ROOT,BLOCK,digest,atomic_json,csv_write,record,events,index
from core import apply_pattern,match_pattern


def run_worker(run,seed,smoke=False):
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    sys.path.insert(0,str(ROOT/'sdxl-unbox'))
    from SAE import SparseAutoencoder
    from SDLens import HookedStableDiffusionXLPipeline
    cfg=json.loads((run/'config/experiment_config.json').read_text())
    estimate=json.loads((run/'config/storage_estimate.json').read_text())
    if __import__('shutil').disk_usage(run).free<estimate['uncompressed_tensor_bytes']+estimate['png_upper_bytes']:
        raise OSError('Insufficient disk space for conservative full-tensor estimate')
    conditions=[c for c in json.loads((run/'config/conditions.json').read_text()) if int(c['prompt_id'].rsplit('seed',1)[1])==seed]
    if smoke:conditions=[c for c in conditions if c['donor_id'] in ('none','shared',cfg['donor_ids'][0])]
    checkpoint=Path(cfg['sae_checkpoint'])
    assert digest(checkpoint/'state_dict.pth')==cfg['sae_sha256']
    assert cfg['layer']==BLOCK and cfg['settings']['steps']==1
    for p,v in cfg['software_versions'].items():
        if metadata.version(p)!=v:raise ValueError('Generation software version changed: '+p)
    for entry in json.loads((run/'config/input_manifest.json').read_text()):
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Frozen input changed: '+entry['path'])
    for entry in json.loads((run/'config/patterns.json').read_text()):
        if digest(run/entry['path'])!=entry['sha256']:raise ValueError('Pattern changed')
    for entry in __import__('csv').DictReader((run/'config/masks.csv').open()):
        if digest(run/entry['mask_path'])!=entry['mask_sha256']:raise ValueError('Mask changed')
    worker=run/'workers'/f'seed_{seed}'
    for d in ('logs','results','images','activations'):(worker/d).mkdir(parents=True,exist_ok=True)
    with (worker/'logs/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        latest=index(worker)
        baseline=json.loads((run/'config/baselines.json').read_text());prompts={p['sample_id']:p for p in cfg['prompts']}
        settings=cfg['settings'];torch.set_num_threads(4)
        load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
        vae=AutoencoderKL.from_pretrained(settings['vae'],**load)
        pipe=HookedStableDiffusionXLPipeline.from_pretrained(cfg['model_checkpoint'],vae=vae,variant='fp16',**load)
        pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing')
        pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
        sae=SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval();sae.requires_grad_(False)
        atomic_json(worker/'results/runtime.json',dict(device=torch.cuda.get_device_name(),torch=torch.__version__,scheduler=dict(pipe.pipe.scheduler.config),
            precision='diffusion and original block fp16; SAE, coefficients, applied perturbation fp32'))
        # Replay no-op controls before edits; failed pixel agreement stops this worker.
        conditions.sort(key=lambda c:(c['method'] not in ('baseline','no_op'),c['prompt_id'],c['method'],c['pattern_id'],c['target_norm']))
        for condition in conditions:
            cid=condition['image_id'];prior=latest.get(cid)
            if prior and prior['generation_status']=='complete':
                for name in ('image','activation'):
                    if digest(prior[name+'_path'])!=prior[name+'_sha256']:raise ValueError('Completed artifact changed: '+prior[name+'_path'])
                continue
            prompt=prompts[condition['prompt_id']];b=baseline[condition['prompt_id']]
            if condition['method']=='baseline':
                row=dict(condition,run_id=cfg['run_id'],image_path=str(run/b['image_path']),activation_path=str(run/b['activation_path']),
                    generation_status='complete',image_sha256=b['image_sha256'],activation_sha256=b['activation_sha256'],
                    prompt_text=prompt['prompt'],seed=seed,group=b['group'],feature_id=4228,layer=BLOCK,
                    model_checkpoint=cfg['model_checkpoint'],sae_checkpoint=cfg['sae_checkpoint'],reused=True,
                    original_activation_nonzero_fraction=b['nonzero_cells']/256,original_activation_zero=b['exactly_inactive'],
                    actual_applied_norm=0.,match_relative_error=0.,match_passed=True)
                record(worker,row);latest[cid]=row;index(worker);continue
            attempt=sum(e['image_id']==cid and e['generation_status']=='started' for e in events(worker))
            stem=cid+f'_attempt{attempt}'
            image=worker/'images'/(stem+'.png');activation=worker/'activations'/(stem+'.npz')
            row=dict(condition,run_id=cfg['run_id'],prompt_text=prompt['prompt'],seed=seed,group=b['group'],feature_id=4228,
                layer=BLOCK,model_checkpoint=cfg['model_checkpoint'],sae_checkpoint=cfg['sae_checkpoint'],sae_sha256=cfg['sae_sha256'],
                image_path=str(image),activation_path=str(activation),timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                attempt=attempt,generation_status='started',original_activation_zero=b['exactly_inactive'])
            record(worker,row);captured={};stats={}
            if condition['method']=='no_op':pattern=np.zeros((16,16),np.float32)
            else:
                with np.load(run/condition['pattern_path']) as p:pattern=p['pattern'].copy()
            @torch.no_grad()
            def hook(module,inputs,output):
                original=output[0];residual=(original-inputs[0]).permute(0,2,3,1).float();codes=sae.encode(residual)
                z=codes[...,4228:4229];direction=sae.decoder.weight[:,4228:4229]
                p=torch.from_numpy(pattern).to(z).reshape(1,16,16,1)
                if condition['method']=='no_op':coefficient=0.;error=0.;passed=True
                else:coefficient,error,passed=match_pattern(original,z,p,condition['target_norm'],direction,cfg['tolerance_relative'])
                modified,edited,delta=apply_pattern(original,z,p,coefficient,direction)
                codes_modified=codes.clone();codes_modified[...,4228:4229]=modified
                post=sae.encode((edited-inputs[0]).permute(0,2,3,1).float())
                applied=edited.float()-original.float();change=modified-z
                tensors=dict(original_feature=z,modified_feature=modified,post_reencoded_feature=post[...,4228:4229],
                    original_latents=codes,modified_latents=codes_modified,original_block=original,
                    applied_block_delta=applied,precast_block_delta=delta.permute(0,3,1,2),decoder_direction=direction,
                    base_pattern=p,coefficient=torch.tensor(coefficient),latent_change=change,
                    scheduler_timestep=pipe.pipe.scheduler.timesteps[0],feature_id=torch.tensor(4228))
                for name,value in tensors.items():captured[name]=value.detach().cpu().numpy()
                captured['layer']=np.asarray(BLOCK)
                tmp=activation.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**captured);f.flush();os.fsync(f.fileno())
                os.replace(tmp,activation)
                stats.update(coefficient=coefficient,actual_applied_norm=float(applied.norm()),match_relative_error=error,match_passed=passed,
                    original_activation_mean=float(z.mean()),original_activation_max=float(z.max()),original_activation_nonzero_fraction=float((z!=0).float().mean()),
                    modified_activation_mean=float(modified.mean()),modified_activation_max=float(modified.max()),
                    modified_activation_nonzero_fraction=float((modified!=0).float().mean()),
                    modified_negative_fraction=float((modified<0).float().mean()),perturbation_norm=float(change.norm()),
                    original_mean_support=float((codes!=0).sum(-1).float().mean()),modified_mean_support=float((codes_modified!=0).sum(-1).float().mean()),
                    introduced_support_fraction=float(((z==0)&(modified!=0)).float().mean()),
                    post_reencode_selected_l2=float((post[...,4228:4229]-modified).norm()),
                    relative_full_latent_change=float(change.norm()/codes.norm().clamp_min(1e-12)),
                    denoising_timesteps=[float(pipe.pipe.scheduler.timesteps[0])])
                return output if condition['method']=='no_op' else (edited,)+output[1:]
            try:
                noise=torch.Generator('cuda').manual_seed(seed);scheduler=torch.Generator('cuda').manual_seed(seed)
                latents=torch.randn((1,4,64,64),generator=noise,device='cuda',dtype=torch.float16)
                result=pipe.run_with_hooks(prompt=prompt['prompt'],latents=latents,generator=scheduler,num_inference_steps=1,
                    guidance_scale=settings['guidance_scale'],height=512,width=512,position_hook_dict={BLOCK:hook})
                result.images[0].save(image,format='PNG')
                row.update(stats,generation_status='complete',image_sha256=digest(image),activation_sha256=digest(activation))
                if condition['method']=='no_op':
                    with Image.open(image) as im,Image.open(run/b['image_path']) as ref:
                        equal=np.array_equal(np.asarray(im),np.asarray(ref))
                    row['baseline_pixels_equal']=equal
                    if not equal:raise ValueError('No-op pixel replay differs from saved baseline')
                if not np.isfinite(captured['applied_block_delta']).all():raise FloatingPointError('Nonfinite intervention preserved')
            except BaseException as exc:
                row.update(generation_status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=traceback.format_exc())
                record(worker,row);index(worker);raise
            record(worker,row);latest[cid]=row;index(worker)
            print(seed,condition['prompt_id'],condition['method'],condition['donor_id'],condition['target_norm'],stats['match_relative_error'],flush=True)
        selected=[latest[c['image_id']] for c in conditions]
        if not all(r.get('match_passed',False) for r in selected):raise ValueError('Some conditions outside norm tolerance; images retained')
        if smoke:
            edits=[r for r in selected if r['method'] not in ('baseline','no_op')]
            assert len(edits)==24
            assert any(r['original_activation_zero'] for r in edits)
            atomic_json(run/'results/smoke_validation.json',dict(passed=True,seed=seed,edited_conditions=len(edits),
                max_relative_error=max(r['match_relative_error'] for r in edits),no_op_controls=2,
                arrays='raw patterns, original/modified maps, full latent dictionaries and applied block tensors preserved'))
            print('SMOKE PASSED',flush=True)
        else:atomic_json(worker/'results/generation_complete.json',dict(complete=True,conditions=len(conditions)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.smoke:
        if not json.loads((a.run/'results/smoke_validation.json').read_text())['passed']:raise ValueError('Smoke not passed')
    run_worker(a.run.resolve(),a.seed,a.smoke)
if __name__=='__main__':main()
