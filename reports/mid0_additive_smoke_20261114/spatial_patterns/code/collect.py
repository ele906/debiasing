"""Build complete condition indexes, including unstarted/failed rows, from append-only workers."""
import json
import numpy as np
from pathlib import Path
import sys
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity');sys.path.insert(0,str(BASE))
from run import atomic_json,csv_write,events

def collect(run):
    cfg=json.loads((run/'config/experiment_config.json').read_text());latest={};attempts=[]
    for worker in sorted((run/'workers').glob('seed_*')):
        for e in events(worker):attempts.append(dict(e,worker=str(worker)));latest[e['image_id']]=e
    source_times={}
    for e in events(Path(cfg['source_run'])):
        if e.get('intervention_type')=='baseline' and e.get('generation_status')=='complete':source_times[e['prompt_id']]=e.get('timestamp')
    rows=[]
    for c in json.loads((run/'config/conditions.json').read_text()):
        r=latest.get(c['image_id'],dict(c,run_id=cfg['run_id'],generation_status='not_started'))
        r['intervention_type']=r['method'];r['sae_sha256']=cfg['sae_sha256']
        if r['method']=='baseline' and r['generation_status']=='complete':
            r['timestamp']=source_times.get(r['prompt_id'])
            with np.load(r['activation_path']) as archive:
                j=list(archive['feature_ids']).index(4228);z=archive['original_0'][...,j]
                r.update(original_activation_mean=float(z.mean()),original_activation_max=float(z.max()),modified_activation_mean=float(z.mean()),modified_activation_max=float(z.max()),
                    modified_activation_nonzero_fraction=float(np.mean(z!=0)),perturbation_norm=0.,denoising_timesteps=[float(archive['timestep_0'])])
        rows.append(r)
    csv_write(run/'results/all_generations.csv',rows);csv_write(run/'results/all_attempts.csv',attempts)
    atomic_json(run/'results/all_generations.json',rows)
    return rows
