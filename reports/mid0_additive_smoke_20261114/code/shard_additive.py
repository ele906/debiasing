"""Fork completed calibration/controls into independent feature workers using immutable hardlinks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
from run import atomic_json,index,record,inventory,csv_write,digest

p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
parent=a.run.resolve()
with (parent/'logs/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((parent/'config/experiment_config.json').read_text())
 assert json.loads((parent/'results/smoke_validation.json').read_text())['passed']
 latest=index(parent)
 workers=[]
 for feature in cfg['features']:
  worker=parent/'workers'/f'feature_{feature}'
  worker.mkdir(parents=True,exist_ok=False)
  for d in ('config/source','images','activations/raw','activations/modified','results','evaluations','logs','figures'):
   (worker/d).mkdir(parents=True,exist_ok=True)
  child=dict(cfg,run_id=parent.name+f'_feature_{feature}',worker_features=[feature],shared_smoke=str(parent/'results/smoke_validation.json'),shared_parent=str(parent))
  for path in (parent/'config').glob('*'):
   if path.is_file():shutil.copy2(path,worker/'config'/path.name)
  for path in (parent/'config/source').glob('*'):
   if path.is_file():shutil.copy2(path,worker/'config/source'/path.name)
  # Record the parallel implementation as a new source artifact; original snapshots remain preserved in the parent.
  paths=[]
  for path in Path(__file__).parent.glob('*.py'):
   target=worker/'config/source'/path.name;shutil.copy2(path,target)
   paths.append(dict(path=str(target),resolved_path=str(target.resolve()),bytes=target.stat().st_size,sha256=digest(target)))
  inputs=json.loads((parent/'config/checkpoint_manifest.json').read_text())
  inputs=[x for x in inputs if '/config/source/' not in x['path']]+paths
  atomic_json(worker/'config/checkpoint_manifest.json',inputs)
  atomic_json(worker/'config/experiment_config.json',child)
  reused=0
  for old in latest.values():
   if old['generation_status']!='complete' or old['feature_id'] not in (-1,feature):continue
   r=dict(old,run_id=child['run_id'],reused_from=str(parent),reuse_source_image=str(parent/old['image_path']))
   for name in ('image','activation'):
    source=parent/old[name+'_path'];target=worker/old[name+'_path'];target.parent.mkdir(parents=True,exist_ok=True)
    assert digest(source)==old[name+'_sha256']
    os.link(source,target)
   record(worker,r);reused+=1
  index(worker)
  (worker/'README.md').write_text(f'Parallel additive worker for feature {feature}. Shared parent: {parent}.\nResume with additive.py --run {worker}.\nCompleted shared controls are immutable hardlinks; no image or activation recomputation. Calibration and all 10 selected-feature maps use the same parent config.\n')
  workers.append(dict(feature_id=feature,run=str(worker),reused_conditions=reused))
 atomic_json(parent/'config/parallel_workers.json',workers)
 csv_write(parent/'results/parallel_workers.csv',workers)
 print(json.dumps(workers,indent=2))
