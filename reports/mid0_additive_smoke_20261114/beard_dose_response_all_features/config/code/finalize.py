"""Add concise findings/checks and update hashes only for new or changed outputs."""
import argparse,csv,json,shutil
from pathlib import Path
from prepare import sha,write,read
from findings import main as findings_main
from refresh_comparison_plots import main as refresh_plots

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
 refresh_plots()
 findings_main()
 prior={r['path']:r for r in read(run/'output_manifest.csv')};rows=[]
 mutable={'README.md','FINDINGS.md','results/integrity_checks.json','results/feature4228_selected_effects.csv'}
 for path in sorted(run.rglob('*')):
  if not path.is_file() or path.name=='output_manifest.csv' or path.suffix=='.lock' or '__pycache__' in path.parts:continue
  rel=str(path.relative_to(run));old=prior.get(rel)
  if old and int(old['bytes'])==path.stat().st_size and rel not in mutable and not rel.startswith('config/code/') and not rel.startswith('figures/ten_feature_'):
   rows.append(old)
  else:rows.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path)))
 write(run/'output_manifest.csv',rows);print('Finalized',run,len(rows),'hashed files',flush=True)
if __name__=='__main__':main()
