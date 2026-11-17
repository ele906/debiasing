"""Publish only derived analysis outputs through isolated Git index plumbing."""
import argparse,csv,gzip,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
from prepare import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();run=a.run.resolve();repo=Path('/tmp/mid0-additive-github-20261114');rel=Path('reports/mid0_additive_smoke_20261114')/a.name;dst=repo/rel
 assert json.loads((run/'results/complete.json').read_text())['n_new_generated_images']==0
 dst.mkdir(exist_ok=True);shutil.copy2(run/'README.md',dst/'README.md');shutil.copytree(run/'figures',dst/'figures',dirs_exist_ok=True)
 (dst/'results').mkdir(exist_ok=True);(dst/'config/code').mkdir(parents=True,exist_ok=True)
 for source in (run/'results').iterdir():
  if not source.is_file() or source.name.startswith('private_'):continue
  if source.stat().st_size>5_000_000 or source.name in ['per_image_scores.csv','common_seed_comparison.csv']:
   with source.open('rb') as inp,(dst/'results'/(source.name+'.gz')).open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:shutil.copyfileobj(inp,out)
  else:shutil.copy2(source,dst/'results'/source.name)
 for source in (run/'config').iterdir():
  if not source.is_file() or source.name.startswith('private_') or source.name=='git_status.txt':continue
  if source.name in ['conditions.csv','assets.csv']:
   with source.open('rb') as inp,(dst/'config'/(source.name+'.gz')).open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:shutil.copyfileobj(inp,out)
  else:shutil.copy2(source,dst/'config'/source.name)
 for source in (run/'config/code').iterdir():
  if source.is_file() and source.suffix in ['.py','.sbatch']:shutil.copy2(source,dst/'config/code'/source.name)
 shutil.copy2(run/'output_manifest.csv',dst/'local_full_output_manifest.csv')
 with (dst/'README.md').open('a') as f:f.write('\n\n## GitHub release scope\n\nLarge per-image tables and input manifests are losslessly gzip-compressed (`.csv.gz`); read with pandas.read_csv or gzip.open. Private per-case distortion-review joins and reviewer labels are excluded. The full masks/probabilities/originals remain in `'+str(run)+'`. The local full-output manifest lists those files; they are not all uploaded here.\n')
 write(dst/'published_manifest.csv',[dict(path=str(p.relative_to(dst)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dst.rglob('*')) if p.is_file() and p.name!='published_manifest.csv'])
 assert all(p.stat().st_size<95_000_000 for p in dst.rglob('*') if p.is_file())
 def git(*args,env=None):return subprocess.check_output(['git','-C',str(repo),*args],env=env,text=True).strip()
 head=git('rev-parse','HEAD');env=dict(os.environ,GIT_INDEX_FILE=tempfile.mktemp(prefix='beard-publish-',suffix='.index',dir='/tmp'));git('read-tree',head,env=env)
 for file in sorted(dst.rglob('*')):
  if file.is_file():git('update-index','--add','--cacheinfo','100644',git('hash-object','-w',str(file)),str(file.relative_to(repo)),env=env)
 tree=git('write-tree',env=env);changes=git('diff-tree','--no-commit-id','--name-status','-r',head,tree)
 if not changes:print('Already published');return
 assert all(l.split('\t')[-1].startswith(str(rel)+'/') and not l.startswith('D\t') for l in changes.splitlines())
 commit=git('commit-tree',tree,'-p',head,'-m','Score existing SAE interventions with beard parser: '+a.name);git('update-ref','refs/heads/mid0-additive-smoke-20261114',commit,head)
 subprocess.run(['git','-C',str(repo),'push','origin','mid0-additive-smoke-20261114'],check=True)
 print('PUBLISHED',commit,'https://github.com/ele906/debiasing/tree/mid0-additive-smoke-20261114/'+str(rel),flush=True)
if __name__=='__main__':main()
