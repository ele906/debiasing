"""Freeze patterns, verified baselines, provenance and the full condition plan."""
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,BLOCK,digest,atomic_json,csv_write
from additive import CHECKPOINT,EXPECTED_HASH
BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'additive_20261114_174738_954258'


def main():
    assert digest(CHECKPOINT/'state_dict.pth')==EXPECTED_HASH
    prior=json.loads((SOURCE/'config/experiment_config.json').read_text())
    latest={}
    for line in (SOURCE/'logs/events.jsonl').read_text().splitlines():
        try:r=json.loads(line);latest[r['image_id']]=r
        except ValueError:pass
    prompts=[p for p in prior['prompts'] if p['race_prompt'] in ('black','east asian') and p['gender_prompt']=='man']
    run=BASE/('spatial4228_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    for d in ('config/source','patterns/donors','patterns/shuffled','patterns/uniform','masks','images/baseline','activations/baseline','results','evaluations','figures','logs','workers'):
        (run/d).mkdir(parents=True,exist_ok=True)
    cfg=dict(run_id=run.name,feature_id=4228,layer=BLOCK,sae_checkpoint=str(CHECKPOINT.resolve()),sae_sha256=EXPECTED_HASH,
        model_checkpoint=prior['model_checkpoint'],settings=prior['settings'],software_versions=prior['software_versions'],prompts=prompts,
        donor_ids=['black_man_seed030','black_man_seed031'],shuffle_seeds=[1729,2718,3141],targets=[42.5242,65.1363],
        tolerance_relative=.005,operation='z_i + lambda*P; residual-preserving block-output delta; no clipping or Top-K reapplication',
        mask_procedure='Existing MediaPipe lower-face extended-chin soft polygon; 512px mask with Gaussian blur 2; Pillow mode-F BOX area averaging to 16x16; fractional mask retained',
        localized_definition='Uniform activation weighted by recipient baseline lower-face mask; shared across donors',
        alignment='Raw donor grid copied in absolute image coordinates; no warping or landmark alignment of donor pattern',
        scientific_status='Exploratory: no validated beard semantics assumed; seed33 East-Asian recipient has nonzero cells',
        source_run=str(SOURCE),full_latents=True)
    context=[];baselines={}
    for prompt in prompts:
        r=next(r for r in latest.values() if r['intervention_type']=='baseline' and r['prompt_id']==prompt['sample_id'] and r['generation_status']=='complete')
        for kind,folder in [('image','images/baseline'),('activation','activations/baseline')]:
            src=SOURCE/r[kind+'_path'];assert digest(src)==r[kind+'_sha256']
            dst=run/folder/(prompt['sample_id']+src.suffix);os.link(src,dst)
        archive=run/'activations/baseline'/(prompt['sample_id']+'.npz')
        with np.load(archive) as data:
            j=list(data['feature_ids']).index(4228);z=data['original_0'][...,j].astype(np.float32)
            assert z.shape==(1,16,16)
        c=dict(prompt_id=prompt['sample_id'],seed=int(prompt['seed']),group=r['group'],nonzero_cells=int(np.count_nonzero(z)),
               exactly_inactive=not bool(np.any(z)),max_activation=float(z.max()),image_path=f'images/baseline/{prompt["sample_id"]}.png',
               activation_path=f'activations/baseline/{prompt["sample_id"]}.npz',image_sha256=r['image_sha256'],activation_sha256=r['activation_sha256'])
        context.append(c);baselines[c['prompt_id']]=c
    csv_write(run/'config/context_activity.csv',context)
    atomic_json(run/'config/baselines.json',baselines)
    pattern_rows=[]
    for donor in cfg['donor_ids']:
        with np.load(run/baselines[donor]['activation_path']) as data:
            j=list(data['feature_ids']).index(4228);z=data['original_0'][0,...,j].astype(np.float32)
        assert z.shape==(16,16) and np.any(z)
        path=run/'patterns/donors'/(donor+'.npz')
        np.savez_compressed(path,pattern=z,feature_id=np.asarray(4228),layer=np.asarray(BLOCK))
        pattern_rows.append(dict(pattern_id=donor,method='natural',donor_id=donor,shuffle_seed=None,path=str(path.relative_to(run)),sha256=digest(path)))
        for seed in cfg['shuffle_seeds']:
            rng=np.random.default_rng(seed);permutation=rng.permutation(256)
            shuffled=z.ravel()[permutation].reshape(16,16)
            assert np.array_equal(np.sort(z.ravel()),np.sort(shuffled.ravel()))
            path=run/'patterns/shuffled'/f'{donor}_shuffle{seed}.npz'
            np.savez_compressed(path,pattern=shuffled,permutation=permutation,shuffle_seed=np.asarray(seed))
            pattern_rows.append(dict(pattern_id=f'{donor}_shuffle{seed}',method='shuffled',donor_id=donor,shuffle_seed=seed,path=str(path.relative_to(run)),sha256=digest(path)))
    path=run/'patterns/uniform/ones.npz';np.savez_compressed(path,pattern=np.ones((16,16),np.float32))
    pattern_rows.append(dict(pattern_id='uniform',method='uniform',donor_id='shared',shuffle_seed=None,path=str(path.relative_to(run)),sha256=digest(path)))
    atomic_json(run/'config/patterns.json',pattern_rows)
    conditions=[]
    for prompt in prompts:
        pid=prompt['sample_id']
        specs=pattern_rows+[dict(pattern_id='localized_'+pid,method='localized',donor_id='shared',shuffle_seed=None,path=f'masks/{pid}.npz',sha256=None)]
        for spec in specs:
            for target in cfg['targets']:
                c=dict(prompt_id=pid,method=spec['method'],pattern_id=spec['pattern_id'],donor_id=spec['donor_id'],shuffle_seed=spec['shuffle_seed'],
                       pattern_path=spec['path'],target_norm=target)
                c['image_id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:24];conditions.append(c)
        for method in ('baseline','no_op'):
            c=dict(prompt_id=pid,method=method,pattern_id='none',donor_id='none',shuffle_seed=None,pattern_path=None,target_norm=0.)
            c['image_id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:24];conditions.append(c)
    assert len(conditions)==220
    atomic_json(run/'config/conditions.json',conditions)
    atomic_json(run/'config/experiment_config.json',cfg)
    files=[CHECKPOINT/'state_dict.pth',CHECKPOINT/'config.json',ROOT/'sdxl-unbox/SAE/sae.py',ROOT/'sdxl-unbox/SDLens/hooked_sd_pipeline.py',
        ROOT/'face-landmarker/render_landmarked_lower_face_change_report.py',ROOT/'face-landmarker/models/face_landmarker.task']
    for p in Path(__file__).parent.glob('*.py'):
        dst=run/'config/source'/p.name;shutil.copy2(p,dst);files.append(dst)
    for p in (BASE/'run.py',BASE/'additive.py'):
        dst=run/'config/source'/p.name;shutil.copy2(p,dst);files.append(dst)
    atomic_json(run/'config/input_manifest.json',[dict(path=str(p),resolved_path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)) for p in files])
    # Keep the previously recorded local diffusion snapshot hashes without confusing them with SAE provenance.
    old=json.loads((SOURCE/'config/checkpoint_manifest.json').read_text())
    atomic_json(run/'config/generator_checkpoint_manifest.json',[r for r in old if '/ReNO/hf_cache/' in r['path']])
    env=sys.version+'\n'
    for cmd in (['git','rev-parse','HEAD'],['git','status','--porcelain'],[sys.executable,'-m','pip','freeze']):
        env+='\n'+repr(cmd)+'\n'+subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True).stdout
    (run/'config/environment.txt').write_text(env)
    estimate=220*(2*5120+1280*3)*256*4
    atomic_json(run/'config/storage_estimate.json',dict(uncompressed_tensor_bytes=estimate,png_upper_bytes=220*512*512*4,free_bytes=shutil.disk_usage(run).free))
    print(run,flush=True)

if __name__=='__main__':main()
