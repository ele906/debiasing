"""Prepare signed feature-4228 fixed-value tests; reuse only verified matching controls."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
from run import atomic_json,index,record,digest,csv_write

base=Path(__file__).parent.resolve()
source=base/'additive_20261114_174738_954258'
cfg=json.loads((source/'config/experiment_config.json').read_text())
values=[-200.,-150.,-100.,-50.,-25.,-10.,0.,10.,25.,50.,100.,150.,200.]
prompts=[p for p in cfg['prompts'] if p['race_prompt'] in ('black','east asian') and p['gender_prompt']=='man']
assert len(prompts)==10
run=base/('fixed4228_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
run.mkdir(exist_ok=False)
for d in ('config','results','figures','logs','images','activations/raw','activations/modified','evaluations'):(run/d).mkdir(parents=True,exist_ok=True)
full=dict(cfg,run_id=run.name,prompts=prompts,features=[4228],worker_features=[4228],coordinate_operation='set',set_values=values,
          fixed_offsets=[],shared_smoke=str(source/'results/smoke_validation.json'),feature_selection='User hypothesis: candidate 4228 may control beard appearance; test rather than assume',
          note='Set all selected-feature spatial coordinates to target, including negative out-of-distribution values; preserve residual; no clipping')
atomic_json(run/'config/experiment_config.json',full)
source_rows=index(source)
workers=[]
for seed in range(30,35):
 worker=run/'workers'/f'seed_{seed}'
 for d in ('config/source','results','figures','logs','images','activations/raw','activations/modified','evaluations'):(worker/d).mkdir(parents=True,exist_ok=True)
 child=dict(full,run_id=run.name+f'_seed_{seed}',prompts=[p for p in prompts if int(p['seed'])==seed])
 for path in (source/'config').glob('*'):
  if path.is_file():shutil.copy2(path,worker/'config'/path.name)
 entries=[e for e in json.loads((source/'config/checkpoint_manifest.json').read_text()) if '/config/source/' not in e['path']]
 for path in base.glob('*.py'):
  target=worker/'config/source'/path.name;shutil.copy2(path,target)
  entries.append(dict(path=str(target),resolved_path=str(target.resolve()),bytes=target.stat().st_size,sha256=digest(target)))
 atomic_json(worker/'config/checkpoint_manifest.json',entries)
 atomic_json(worker/'config/experiment_config.json',child)
 atomic_json(worker/'config/storage_estimate.json',dict(conditions=30,activation_bytes_uncompressed=30*16*16*5120*2*4,png_upper_estimate_bytes=30*512*512*4))
 ids={p['sample_id'] for p in child['prompts']}
 for old in source_rows.values():
  if old['generation_status']!='complete' or old['prompt_id'] not in ids or old['feature_id']!=-1:continue
  r=dict(old,run_id=child['run_id'],reused_from=str(source))
  for name in ('image','activation'):
   origin=source/old[name+'_path'];target=worker/old[name+'_path'];target.parent.mkdir(parents=True,exist_ok=True)
   assert digest(origin)==old[name+'_sha256'];os.link(origin,target)
  record(worker,r)
 index(worker)
 workers.append(dict(seed=seed,run=str(worker)))
atomic_json(run/'config/parallel_workers.json',workers)
(run/'README.md').write_text(f'''# Fixed activation sweep: feature 4228\n\nReplace z_4228 at every 16x16 spatial cell by each signed target: {values}. This is coordinate replacement, not multiplication or a constant additive offset. All other latents and the SAE reconstruction residual are preserved.\n\nPrompts: East-Asian-man and Black-man, seeds 30–34. Checkpoint: layers/runs/mid0_sae_face_001/sae, SHA-256 {cfg['sae_sha256']}. Diffusion, scheduler, precision, RNG, prompts and seeds match the existing sweep. Negative values cannot arise from the natural ReLU/Top-K encoder and are explicitly flagged. High positive uniform targets also depart from natural sparse support. All images, full latent tensors, post-reencoding maps and applied perturbation norms are retained in workers/.\n\n5 independent GPU workers each produce 26 edited images, reusing 4 verified controls. Existing additive smoke validation establishes pipeline compatibility; fixed-value validity is checked separately from the saved exact modified coordinates. Do not interpret the prior additive smoke pass as validation of the biological or semantic hypothesis.\n''')
print(run,flush=True)
