"""Additive companion to active stability_top10_4243081; never edits its outputs."""
import argparse
import datetime as dt
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import csv

from run import ROOT, BLOCK, digest, atomic_json, csv_write, events, record, index, inventory

CHECKPOINT = ROOT/'layers/runs/mid0_sae_face_001/sae'
EXPECTED_HASH = 'cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6'
SOURCE_RUN = ROOT/'layers/runs/mid0_sae_face_001/interventions/stability_top10_4243081'
RANKING = ROOT/'layers/mid0_stability/finetuned_ranking/top10.json'


def create_run(args):
    ranking=json.loads(RANKING.read_text())
    assert ranking['checkpoint_sha256']==EXPECTED_HASH
    assert digest(CHECKPOINT/'state_dict.pth')==EXPECTED_HASH
    with (ROOT/'backprop/manifest.csv').open() as f:
        prompts=[r for r in csv.DictReader(f) if 30<=int(r['seed'])<=34]
    assert len(prompts)==60
    run=Path(__file__).parent/('additive_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(exist_ok=False)
    for name in ('config/source','images','activations/raw','activations/modified','results','evaluations','logs','figures'):
        (run/name).mkdir(parents=True,exist_ok=True)
    cfg=dict(run_id=run.name,layer=BLOCK,model_checkpoint='stabilityai/sdxl-turbo',sae_checkpoint=str(CHECKPOINT.resolve()),
             sae_sha256=EXPECTED_HASH,multiplicative_run=str(SOURCE_RUN),features=ranking['features'],prompts=prompts,
             feature_selection=ranking['ranking'],settings=json.loads((ROOT/'backprop/generation_settings.json').read_text()),
             offsets=dict(small='0.1 * pooled positive activation q25',moderate='0.5 * pooled positive activation q50'),
             match_targets=dict(matched_small='q25 active-context ||z_i||_2 * ||decoder_i||_2',
                                matched_moderate='q50 active-context ||z_i||_2 * ||decoder_i||_2'),
             fixed_offsets=[10.,20.,50.],match_tolerance_relative=.005,full_tensors=not args.selected_only,
             software_versions={p:metadata.version(p) for p in ('torch','diffusers','transformers','numpy','Pillow')},
             checkpoint_provenance='face-region fine-tuned SAE; diffusion generator frozen',
             semantics='Perceived appearance predictions, not ground-truth demographic identities')
    atomic_json(run/'config/experiment_config.json',cfg)
    atomic_json(run/'config/prompts.json',prompts)
    atomic_json(run/'config/feature_selection.json',ranking)
    csv_write(run/'config/feature_list.csv',[dict(feature_id=f,checkpoint_sha256=EXPECTED_HASH,selection=ranking['ranking']) for f in cfg['features']])
    paths=[CHECKPOINT/'state_dict.pth',CHECKPOINT/'config.json',RANKING,ROOT/'backprop/manifest.csv',
           ROOT/'backprop/generation_settings.json',ROOT/'layers/mid0_stability/generate_sweep.py',
           ROOT/'layers/mid0_stability/generate_top10.sbatch',ROOT/'sdxl-unbox/SAE/sae.py',ROOT/'sdxl-unbox/SDLens/hooked_sd_pipeline.py']
    for p in Path(__file__).parent.glob('*.py'):
        copy=run/'config/source'/p.name;shutil.copy2(p,copy);paths.append(copy)
    for name in ('stabilityai--sdxl-turbo','madebyollin--sdxl-vae-fp16-fix'):
        paths.extend(p for p in (ROOT/'ReNO/hf_cache'/('models--'+name)/'snapshots').glob('**/*') if p.is_file())
    atomic_json(run/'config/checkpoint_manifest.json',[dict(path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)) for p in paths])
    env=sys.version+'\n'
    for cmd in (['git','rev-parse','HEAD'],['git','status','--porcelain'],[sys.executable,'-m','pip','freeze']):
        env+='\n'+repr(cmd)+'\n'+subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True).stdout
    (run/'config/environment.txt').write_text(env)
    # 16x16 positions, 5,120 fp32 coordinates, 2 full arrays; selected maps and images additional.
    estimate=(60+4200+60+4)*16*16*(5120*2 if cfg['full_tensors'] else 30)*4
    atomic_json(run/'config/storage_estimate.json',dict(conditions=4324,activation_bytes_uncompressed=estimate,
        png_upper_estimate_bytes=4324*512*512*4,free_bytes=shutil.disk_usage(run).free,
        note='Full original/modified dictionaries optional; all selected feature maps always retained; estimates for one-step 512px generation'))
    (run/'README.md').write_text(f'''# Additive companion experiment\n\nResume from /n/fs/goose:\n\n```bash\nexport PYTHONNOUSERSITE=1\nexport PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps\n/n/fs/goose/el8403/conda-envs/goose/bin/python experiments/mid0_sae_selectivity/additive.py --run {run}\n```\n\nCheckpoint: `{CHECKPOINT}`, SHA-256 `{EXPECTED_HASH}`. Matches multiplicative run `{SOURCE_RUN}`: 10 features, 12 prompt groups, seeds 30–34. No old epoch-8 pilot outputs are used.\n\nOriginal PNGs live in images/. NPZ archives in activations/raw/ retain per-hook original and modified selected-feature maps, post-reencoding maps, applied offsets, scheduler timesteps, and optional full original/modified dictionaries. Arrays remain fp32; PNGs remain 512px. All attempts are journaled in logs/events.jsonl; results/all_generations.csv indexes latest condition states. output_manifest.json records file hashes. Existing completed artifacts are verified before skip; every retry has a new filename.\n\nBaseline latents across all 60 contexts calibrate small (0.1×positive q25) and moderate (0.5×positive median) positive offsets. The fixed stronger sweep additionally uses beta=10,20,50 after the smaller conditions. Matching targets are q25/q50 of active-context natural decoder-contribution L2. The same additive function is used in ordinary and matched conditions; matching solves for beta against the actual fp16 applied output norm with relative tolerance 0.5%. Failed matching is flagged. No clipping or reapplication of Top-K occurs.\n\nTop-K/ReLU encoding is nonnegative; linear decoding permits signed and newly introduced coordinates. Positive additions to zero coordinates increase support beyond k=10. Full-support counts, negative fractions, and post-reencoding maps are retained to measure this distribution shift. This intervention is not assumed on-distribution.\n\nThe no-intervention hook returns original output. Beta-zero control checks exact no-op; reconstruction-only substitutes input+decoder(z)+pre_bias. Pixel equality with the multiplicative alpha-1 control is checked before interpreting paired comparisons. Missing source controls are flagged and retried by verification, never assumed equal.\n\nSmoke conditions use two prompt groups and seeds 30,31, including an exactly zero and nonzero context whenever available. The full grid starts only after storage, no-op, pixel-agreement, and active/inactive smoke checks pass. Calibration and smoke selection are exploratory; do not treat them as held-out inference.\n''')
    print(run,flush=True)
    return run


def apply_additive(original,z,beta,direction):
    """Same primitive for ordinary, matched, and zero controls; preserve historical residual."""
    modified=z+beta
    delta=(modified-z)@direction.T
    edited=original+delta.permute(0,3,1,2).to(original)
    return modified,edited,delta


def match_offset(original,z,target,direction):
    import torch
    expected=target/(z.numel()**.5*float(direction.norm()))
    lo=0.;hi=max(expected*2,1e-6)
    def norm(beta):
        _,edited,_=apply_additive(original,z,beta,direction)
        return float((edited.float()-original.float()).norm())
    for _ in range(20):
        if norm(hi)>=target:break
        hi*=2
    best=hi;error=abs(norm(hi)-target)
    for _ in range(32):
        mid=(lo+hi)/2;n=norm(mid)
        if abs(n-target)<error:best,error=mid,abs(n-target)
        if n<target:lo=mid
        else:hi=mid
    return best,error/max(target,1e-12)


def execute(run,smoke_only=False):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    sys.path.insert(0,str(ROOT/'sdxl-unbox'))
    from SAE import SparseAutoencoder
    from SDLens import HookedStableDiffusionXLPipeline
    cfg=json.loads((run/'config/experiment_config.json').read_text())
    if digest(CHECKPOINT/'state_dict.pth')!=cfg['sae_sha256']:raise ValueError('Checkpoint changed')
    versions={p:metadata.version(p) for p in cfg['software_versions']}
    if versions!=cfg['software_versions']:raise ValueError('Software version mismatch on resume')
    for entry in json.loads((run/'config/checkpoint_manifest.json').read_text()):
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Input changed: '+entry['path'])
    estimate=json.loads((run/'config/storage_estimate.json').read_text())
    if shutil.disk_usage(run).free<estimate['activation_bytes_uncompressed']+estimate['png_upper_estimate_bytes']:
        raise OSError('Insufficient estimated storage; explicitly select smaller tensor mode in a NEW run')
    settings=cfg['settings'];features=cfg['features'];prompts=cfg['prompts']
    if settings['steps']!=1:raise ValueError('Companion matching currently validated for exact one-step sweep only')
    torch.set_num_threads(4)
    load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
    vae=AutoencoderKL.from_pretrained(settings['vae'],**load)
    pipe=HookedStableDiffusionXLPipeline.from_pretrained(cfg['model_checkpoint'],vae=vae,variant='fp16',**load)
    pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing')
    pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
    sae=SparseAutoencoder.load_from_disk(str(CHECKPOINT)).float().cuda().eval();sae.requires_grad_(False)
    atomic_json(run/'config/runtime.json',dict(scheduler=dict(pipe.pipe.scheduler.config),device=torch.cuda.get_device_name(),
        precision='diffusion fp16, SAE fp32',sae=json.loads((CHECKPOINT/'config.json').read_text()),versions=versions))
    latest=index(run)

    def generate(prompt,feature,method,label,strength=0.):
        condition=dict(prompt_id=prompt['sample_id'],feature_id=feature,method=method,label=label)
        import hashlib
        image_id=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()[:24]
        if image_id in latest and latest[image_id]['generation_status']=='complete':
            r=latest[image_id]
            for k in ('image','activation'):
                if digest(run/r[k+'_path'])!=r[k+'_sha256']:raise ValueError('Completed artifact changed: '+r[k+'_path'])
            return r
        attempt=sum(e['image_id']==image_id and e['generation_status']=='started' for e in events(run))
        stem=image_id+f'_attempt{attempt}'
        image=run/'images'/method/(f'feature_{feature}' if feature>=0 else 'shared')/(stem+'.png')
        image.parent.mkdir(parents=True,exist_ok=True)
        archive=run/'activations/raw'/(stem+'.npz')
        r=dict(run_id=run.name,image_id=image_id,image_path=str(image.relative_to(run)),activation_path=str(archive.relative_to(run)),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),model_checkpoint=cfg['model_checkpoint'],sae_checkpoint=str(CHECKPOINT),
            layer=BLOCK,feature_id=feature,prompt_id=prompt['sample_id'],prompt_text=prompt['prompt'],seed=int(prompt['seed']),
            group=prompt['race_prompt'].replace(' ','_')+'_'+prompt['gender_prompt'],intervention_type=method,strength_label=label,
            multiplier='',additive_offset=None,original_activation_mean=None,original_activation_max=None,
            original_activation_nonzero_fraction=None,modified_activation_mean=None,perturbation_norm=None,
            denoising_timesteps=[],generation_status='started',attempt=attempt)
        record(run,r)
        captured={};hook_stats=[]
        @torch.no_grad()
        def hook(module,inputs,output):
            original=output[0];residual=(original-inputs[0]).permute(0,2,3,1).float();codes=sae.encode(residual)
            selected=codes[...,features];modified=codes.clone();edited=original
            beta=0.;error=0.;target=None;delta=torch.zeros_like(residual)
            if feature>=0:
                z=codes[...,feature:feature+1];direction=sae.decoder.weight[:,feature:feature+1]
                if method=='matched_additive':
                    target=strength;beta,error=match_offset(original,z,target,direction)
                else:beta=strength
                changed,edited,delta=apply_additive(original,z,beta,direction)
                modified[...,feature:feature+1]=changed
            if method=='reconstruction_only':edited=inputs[0]+(sae.decoder(codes)+sae.pre_bias).permute(0,3,1,2).to(original)
            post=sae.encode((edited-inputs[0]).permute(0,2,3,1).float())
            i=len(hook_stats)
            values=dict(original=selected,modified=modified[...,features],post_reencoded=post[...,features],
                        latent_change=modified[...,features]-selected)
            if cfg['full_tensors']:values.update(full_original=codes,full_modified=modified)
            for name,value in values.items():captured[f'{name}_{i}']=value.cpu().numpy()
            captured.update(feature_ids=np.asarray(features),layer=np.asarray(BLOCK))
            captured[f'timestep_{i}']=np.asarray(pipe.pipe.scheduler.timesteps[i].item())
            captured[f'additive_offset_{i}']=np.asarray(beta)
            hook_stats.append(dict(beta=beta,target=target,match_relative_error=error,
                applied_norm=float((edited.float()-original.float()).norm()),precast_norm=float(delta.norm()),
                original_support=float((codes!=0).sum(-1).float().mean()),modified_support=float((modified!=0).sum(-1).float().mean()),
                modified_negative_fraction=float((modified<0).float().mean()),
                decoder_norm=float(sae.decoder.weight[:,feature].norm()) if feature>=0 else None,
                zero_to_nonzero_cells=int(((codes==0)&(modified!=0)).sum())))
            tmp=archive.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**captured);f.flush();os.fsync(f.fileno())
            os.replace(tmp,archive)
            return output if method=='baseline' else (edited,)+output[1:]
        try:
            noise=torch.Generator('cuda').manual_seed(r['seed']);scheduler=torch.Generator('cuda').manual_seed(r['seed'])
            latents=torch.randn((1,4,64,64),generator=noise,device='cuda',dtype=torch.float16)
            result=pipe.run_with_hooks(prompt=r['prompt_text'],latents=latents,generator=scheduler,num_inference_steps=settings['steps'],
                guidance_scale=settings['guidance_scale'],height=settings['height'],width=settings['width'],position_hook_dict={BLOCK:hook})
            result.images[0].save(image,format='PNG')
            j=features.index(feature) if feature>=0 else None
            a=captured['original_0'][...,j] if j is not None else captured['original_0']
            b=captured['modified_0'][...,j] if j is not None else captured['modified_0']
            r.update(original_activation_mean=float(a.mean()),original_activation_max=float(a.max()),
                original_activation_nonzero_fraction=float(np.count_nonzero(a)/a.size),original_activation_zero=bool(not np.any(a)),
                modified_activation_mean=float(b.mean()),modified_activation_max=float(b.max()),
                modified_activation_nonzero_fraction=float(np.count_nonzero(b)/b.size),modified_activation_negative_fraction=float(np.mean(b<0)),
                perturbation_norm=float(np.linalg.norm(b-a)),perturbation_rms=float(np.sqrt(np.mean((b-a)**2))),
                decoded_perturbation_norm=hook_stats[0]['applied_norm'],precast_perturbation_norm=hook_stats[0]['precast_norm'],
                additive_offset=hook_stats[0]['beta'],target_applied_norm=hook_stats[0]['target'],
                match_relative_error=hook_stats[0]['match_relative_error'],match_converged=hook_stats[0]['match_relative_error']<=cfg['match_tolerance_relative'],
                original_support=hook_stats[0]['original_support'],modified_support=hook_stats[0]['modified_support'],
                zero_to_nonzero_cells=hook_stats[0]['zero_to_nonzero_cells'],denoising_timesteps=[float(captured['timestep_0'])],
                generation_status='complete',image_sha256=digest(image),activation_sha256=digest(archive))
            if not np.isfinite(b).all():r['generation_status']='nonfinite_preserved'
        except BaseException as exc:
            r.update(generation_status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=traceback.format_exc())
            record(run,r);index(run);raise
        record(run,r);latest[image_id]=r;index(run)
        print(r['group'],r['seed'],feature,method,label,'complete',flush=True)
        return r

    baselines={}
    for prompt in prompts:
        baselines[prompt['sample_id']]=generate(prompt,-1,'baseline','no_intervention')
    # Confirm paired control pixels using the active sweep, without reading its files as complete until PNG decoding succeeds.
    checks=[]
    for prompt in prompts:
        own=baselines[prompt['sample_id']]
        group=own['group'];seed=own['seed']
        source=SOURCE_RUN/f'feature_{features[0]}'/'images'/group/f'seed{seed:03d}_alpha_p1.png'
        if not source.exists():checks.append(dict(prompt_id=prompt['sample_id'],status='source_pending',source=str(source)));continue
        from PIL import Image
        with Image.open(run/own['image_path']) as a,Image.open(source) as b:
            equal=np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
        checks.append(dict(prompt_id=prompt['sample_id'],status='equal' if equal else 'mismatch',source=str(source),source_sha256=digest(source)))
        if not equal:raise ValueError('No-intervention pixels differ from multiplicative alpha=1: '+str(source))
    csv_write(run/'results/control_pixel_checks.csv',checks)
    if any(c['status']=='source_pending' for c in checks):raise RuntimeError('Multiplicative controls not yet ready; resume after they are available')
    calibration={};activity=[]
    for j,feature in enumerate(features):
        positive=[];norms=[];context=[]
        for prompt in prompts:
            r=baselines[prompt['sample_id']]
            with np.load(run/r['activation_path']) as a:z=a['original_0'][...,j]
            active=bool(np.any(z));positive.extend(z[z>0].tolist())
            norm=float(np.linalg.norm(z))*float(sae.decoder.weight[:,feature].norm())
            if active:norms.append(norm)
            c=dict(feature_id=feature,prompt_id=prompt['sample_id'],group=r['group'],seed=r['seed'],active=active,
                   activation_nonzero_fraction=float(np.count_nonzero(z)/z.size),natural_decoded_contribution_norm=norm)
            context.append(c);activity.append(c)
        if not positive:raise ValueError(f'Feature {feature} has no active calibration contexts')
        calibration[feature]=dict(small=.1*float(np.quantile(positive,.25)),moderate=.5*float(np.median(positive)),
            matched_small=float(np.quantile(norms,.25)),matched_moderate=float(np.median(norms)),
            active_contexts=sum(c['active'] for c in context),inactive_contexts=sum(not c['active'] for c in context))
    atomic_json(run/'config/additive_calibration.json',calibration)
    csv_write(run/'activations/context_activity.csv',activity)
    if cfg.get('shared_smoke'):
        shared=json.loads(Path(cfg['shared_smoke']).read_text())
        if not shared['passed']:raise ValueError('Shared smoke test has not passed')
        if smoke_only:return
    else:
        # Smoke: one selected feature, two groups, two shared seeds. Choose groups only by pre-intervention activity.
        feature=features[0]
        inactive=[c for c in activity if c['feature_id']==feature and not c['active'] and c['seed'] in (30,31)]
        active=[c for c in activity if c['feature_id']==feature and c['active'] and c['seed'] in (30,31)]
        if not inactive or not active:raise ValueError('Smoke requires active and inactive contexts in paired seeds 30,31')
        groups={inactive[0]['group'],active[0]['group']}
        if len(groups)<2:
            groups.add(next(c['group'] for c in activity if c['group'] not in groups))
        smoke_prompts=[p for p in prompts if int(p['seed']) in (30,31) and p['race_prompt'].replace(' ','_')+'_'+p['gender_prompt'] in groups]
        smoke=[]
        def edits(p,f):
            result=[]
            for label in ('small','moderate','matched_small','matched_moderate'):
                result.append(generate(p,f,'matched_additive' if label.startswith('matched') else 'additive',label,calibration[f][label]))
            for beta in cfg.get('fixed_offsets',[]):
                result.append(generate(p,f,'additive',f'fixed_{beta:g}',beta))
            return result
        for prompt in smoke_prompts:
            zero=generate(prompt,feature,'additive','zero_control',0.)
            if zero['image_sha256']!=baselines[prompt['sample_id']]['image_sha256']:raise ValueError('Beta-zero control failed')
            smoke.extend(edits(prompt,feature))
        if not any(r['original_activation_zero'] and r['modified_activation_nonzero_fraction']>0 for r in smoke):
            raise ValueError('No fully inactive context was activated during smoke test')
        if not all(r['match_converged'] for r in smoke):raise ValueError('Matched smoke norm outside tolerance')
        atomic_json(run/'results/smoke_validation.json',dict(passed=True,feature_id=feature,prompt_ids=[p['sample_id'] for p in smoke_prompts],
            image_ids=[r['image_id'] for r in smoke],conditions=len(smoke),inactive_activated=sum(r['original_activation_zero'] for r in smoke),
            control_pixel_matches=len(checks),full_tensors=cfg['full_tensors']))
        print('SMOKE PASSED',flush=True)
        if smoke_only:return
    for prompt in prompts:
        generate(prompt,-1,'reconstruction_only','full_reconstruction')
    for feature in cfg.get('worker_features',features):
        for prompt in prompts:edits(prompt,feature)
    atomic_json(run/'results/generation_complete.json',dict(complete=True,recorded_conditions=len(index(run))))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--selected-only',action='store_true',help='Explicitly omit full dictionary tensors; retain all selected-feature maps')
    args=p.parse_args();run=args.run.resolve() if args.run else create_run(args)
    with (run/'logs/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.prepare_only:
            try:execute(run,args.smoke_only)
            except BaseException:
                with (run/'logs/failures.jsonl').open('a') as f:f.write(json.dumps(dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),error=traceback.format_exc()))+'\n')
                raise
            finally:index(run);inventory(run)

if __name__=='__main__':main()
