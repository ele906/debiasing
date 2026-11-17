"""Paired beard-parser dose-response curves from existing images; exploratory."""
import argparse,csv,itertools,json,math,os,concurrent.futures as cf
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import read,write,sha
MEN=['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man']
WOMEN=[g[:-3]+'woman' for g in MEN]
METRICS=['delta_current_face_pp','delta_fixed_baseline_face_pp','delta_image_pp']
def face(r):return sum(int(r[f'class_{i}_pixels']) for i in [*range(1,12),14])
def number(v):
 try:return float(v) if v not in ['',None] else None
 except (ValueError,TypeError):return None
def interval(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if len(v)<2:return [None,None]
 b=np.random.default_rng(4228).choice(v,(10000,len(v)),replace=True).mean(1);return np.percentile(b,[2.5,97.5]).tolist()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();cfg=json.loads((run/'config/experiment_config.json').read_text());conditions=read(run/'config/conditions.csv');indexed={r['condition_id']:r for r in conditions};cache={}
 for r in read(run/'config/assets.csv'):
  path=run/'assets'/(r['asset_id']+'.json')
  if path.exists():
   c=json.loads(path.read_text())
   if c.get('status')=='complete':
    assert sha(run/c['beard_mask_path'])==c['beard_mask_sha256'];cache[r['asset_id']]=c
 missing=[r for r in conditions if r['asset_id'] not in cache];write(run/'results/missing_conditions.csv',missing or [dict(status='none_missing')])
 if missing:raise RuntimeError(f'{len(missing)} conditions are missing; resume inference before full analysis')
 rows=[]
 for c in conditions:
  r=dict(c);pred=cache[c['asset_id']];control=indexed[c['control_condition_id']];base=cache[control['asset_id']];f=face(pred);bf=face(base)
  current=pred['beard_pixels']/f if f else None;baseline=base['beard_pixels']/bf if bf else None
  r.update({k:v for k,v in pred.items() if k!='asset_id'});r.update(parser_face_pixels=f,baseline_parser_face_pixels=bf,beard_fraction_current_face=current,baseline_beard_fraction_face=baseline,baseline_beard_pixels=base['beard_pixels'],baseline_asset_id=control['asset_id'],
    beard_fraction_fixed_baseline_face=pred['beard_pixels']/bf if bf else None,
    delta_current_face_pp=100*(current-baseline) if current is not None and baseline is not None else None,
    delta_fixed_baseline_face_pp=100*(pred['beard_pixels']-base['beard_pixels'])/bf if bf else None,
    delta_image_pp=100*(pred['beard_fraction_image']-base['beard_fraction_image']),
    baseline_bearded_proxy=baseline is not None and baseline>=.01,
    original_activation_zero=str(c.get('original_activation_zero','')).lower()=='true',
    caution_multiplier=bool(c['method']=='multiplicative' and abs(float(c['multiplier']))>=50),
    attenuation_amount=1-float(c['multiplier']) if c['method']=='multiplicative' and 0<=float(c['multiplier'])<=1 else None,
    intervention_regime=('sign_reversal' if float(c['multiplier'])<0 else 'removal' if float(c['multiplier'])==0 else 'unchanged' if float(c['multiplier'])==1 else 'amplification') if c['method']=='multiplicative' else c['method'])
  rows.append(r)
 # Local-only linkage to private majority distortion reviews; never publish per-case review mapping.
 review=Path('/n/fs/goose/experiments/mid0_sae_selectivity/face_distortion_consensus.csv');private=[]
 if review.exists():
  rr=read(review)
  write(run/'config/private_review_schema.csv',[dict(column=k) for k in rr[0]])
  for r in rr:
   f=r.get('feature',r.get('feature_id',''));g=r.get('prompt_group',r.get('group',''));seed=r.get('seed','');alpha=r.get('alpha','')
   matches=[v for v in rows if v['method']=='multiplicative' and str(v['feature_id'])==str(f) and v['prompt_group']==g and str(v['seed'])==str(seed) and number(v['multiplier'])==number(alpha)]
   if matches:private.append(dict(condition_id=matches[0]['condition_id'],delta_current_face_pp=matches[0]['delta_current_face_pp'],delta_fixed_baseline_face_pp=matches[0]['delta_fixed_baseline_face_pp'],applied_l2=matches[0]['applied_l2'],**r))
  if private:
   write(run/'results/private_review_join.csv',private)
   aggregated=[]
   for f in sorted({r['feature'] for r in private}):
    for rating in ['yes','no']:
     rr=[r for r in private if r['feature']==f and r['majority_rating']==rating]
     vals=[r['delta_current_face_pp'] for r in rr if r['delta_current_face_pp'] is not None]
     aggregated.append(dict(feature_id=f,majority_clear_distortion=rating,n_reviewed=len(rr),n_with_finite_score=len(vals),mean_delta_current_face_pp=float(np.mean(vals)) if len(vals)>=3 else None,mean_absolute_delta_current_face_pp=float(np.mean(np.abs(vals))) if len(vals)>=3 else None,interpretation='Selected review subset only; no extrapolation to unreviewed images; means suppressed for n<3'))
   write(run/'results/review_aggregate.csv',aggregated)
 write(run/'results/per_image_scores.csv',rows)
 # Same-seed bootstrap: average contexts within seed before across-context summaries.
 features=sorted({int(r['feature_id']) for r in rows});summary=[];sensitivity=[]
 for f,method in itertools.product(features,['multiplicative','additive']):
  part=[r for r in rows if int(r['feature_id'])==f and r['method']==method];strengths=sorted({r['strength_label'] for r in part},key=lambda s:float(s[6:]) if s.startswith('alpha_') else s)
  for group,strength,metric,subset in itertools.product(MEN+WOMEN+['all_contexts'],strengths,METRICS,['all','baseline_bearded']):
   rs=[r for r in part if r['strength_label']==strength and (group=='all_contexts' or r['prompt_group']==group) and (subset=='all' or r['baseline_bearded_proxy']) and r[metric] is not None]
   seeds=sorted({int(r['seed']) for r in rs});v=[np.mean([r[metric] for r in rs if int(r['seed'])==s]) for s in seeds];lo,hi=interval(v)
   summary.append(dict(feature_id=f,method=method,prompt_group=group,strength_label=strength,metric=metric,subset=subset,n_conditions=len(rs),n_seed_clusters=len(v),mean=float(np.mean(v)) if v else None,ci95_low=lo,ci95_high=hi,bootstrap_seed=4228,n_bootstrap=10000,unit='within-seed context mean, then seed bootstrap'))
  for group,strength,threshold in itertools.product(MEN,strengths,[0,.005,.01,.02,.05]):
   rs=[r for r in part if r['prompt_group']==group and r['strength_label']==strength and r['baseline_beard_fraction_face'] is not None and r['baseline_beard_fraction_face']>=threshold]
   v=[r['delta_current_face_pp'] for r in rs if r['delta_current_face_pp'] is not None];lo,hi=interval(v)
   sensitivity.append(dict(feature_id=f,method=method,prompt_group=group,strength_label=strength,baseline_face_fraction_threshold=threshold,n_seed_pairs=len(v),mean_delta_current_face_pp=float(np.mean(v)) if v else None,ci95_low=lo,ci95_high=hi))
 write(run/'results/paired_summary.csv',summary);write(run/'results/baseline_threshold_sensitivity.csv',sensitivity)
 common_summary=[]
 for f,method in itertools.product(features,['multiplicative','additive']):
  part=[r for r in rows if int(r['feature_id'])==f and r['method']==method and 30<=int(r['seed'])<=34]
  for group,label in sorted({(r['prompt_group'],r['strength_label']) for r in part}):
   rr=[r for r in part if r['prompt_group']==group and r['strength_label']==label]
   for metric in METRICS:
    v=[r[metric] for r in rr if r[metric] is not None];lo,hi=interval(v)
    common_summary.append(dict(feature_id=f,method=method,prompt_group=group,strength_label=label,metric=metric,n_seed_pairs=len(v),mean=float(np.mean(v)) if v else None,ci95_low=lo,ci95_high=hi,mean_applied_l2=float(np.mean([float(r['applied_l2']) for r in rr if number(r['applied_l2']) is not None]))))
 write(run/'results/common_seed_summary.csv',common_summary)
 missing_metric=[dict(feature_id=f,method=m,prompt_group=g,n_conditions=len(rr),n_missing_current_face=sum(r['delta_current_face_pp'] is None for r in rr),n_missing_fixed_face=sum(r['delta_fixed_baseline_face_pp'] is None for r in rr)) for f,m,g in sorted({(int(r['feature_id']),r['method'],r['prompt_group']) for r in rows}) for rr in [[r for r in rows if int(r['feature_id'])==f and r['method']==m and r['prompt_group']==g]]]
 write(run/'results/metric_availability.csv',missing_metric)
 # Parser initiation: operational crossing, not verified new facial hair. Keep inactive contexts.
 initiation=[]
 for r in rows:
  if r['method']!='additive' or r['baseline_beard_fraction_face'] is None:continue
  for threshold in [.005,.01,.02,.05]:
   initiation.append(dict(feature_id=r['feature_id'],condition_id=r['condition_id'],prompt_group=r['prompt_group'],seed=r['seed'],strength_label=r['strength_label'],additive_offset=r['additive_offset'],applied_l2=r['applied_l2'],original_activation_zero=r['original_activation_zero'],threshold=threshold,baseline_fraction=r['baseline_beard_fraction_face'],edited_fraction=r['beard_fraction_current_face'],baseline_below_threshold=r['baseline_beard_fraction_face']<threshold,parser_threshold_crossing=r['baseline_beard_fraction_face']<threshold and r['beard_fraction_current_face'] is not None and r['beard_fraction_current_face']>=threshold,baseline_exact_zero=r['baseline_beard_pixels']==0))
 write(run/'results/additive_initiation.csv',initiation)
 init_summary=[]
 for f,g,label,t in sorted({(int(r['feature_id']),r['prompt_group'],r['strength_label'],r['threshold']) for r in initiation}):
  rs=[r for r in initiation if int(r['feature_id'])==f and r['prompt_group']==g and r['strength_label']==label and r['threshold']==t and r['baseline_below_threshold']]
  vals=[float(r['parser_threshold_crossing']) for r in rs];lo,hi=interval(vals)
  init_summary.append(dict(feature_id=f,prompt_group=g,strength_label=label,threshold=t,n_eligible=len(rs),n_crossings=sum(r['parser_threshold_crossing'] for r in rs),crossing_fraction=float(np.mean(vals)) if vals else None,ci95_low=lo,ci95_high=hi))
 write(run/'results/additive_initiation_summary.csv',init_summary)
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(run/'figures'/(name+'.'+ext),dpi=200,bbox_inches='tight')
  plt.close(fig)
 # Per-group curves, individual seed traces and paired seed mean/bootstrap interval.
 for f,gender,subset,metric in itertools.product(features,['men','women'],['all','baseline_bearded'],['delta_current_face_pp','delta_fixed_baseline_face_pp']):
  groups=MEN if gender=='men' else WOMEN;fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
  for ax,g in zip(axes.flat,groups):
   rs=[r for r in rows if int(r['feature_id'])==f and r['method']=='multiplicative' and r['prompt_group']==g and (subset=='all' or r['baseline_bearded_proxy']) and r[metric] is not None]
   for seed in sorted({r['seed'] for r in rs}):
    ss=sorted([r for r in rs if r['seed']==seed],key=lambda r:float(r['multiplier']));ax.plot([float(r['multiplier']) for r in ss],[r[metric] for r in ss],color='#4477AA',lw=.6,alpha=.18)
   ss=sorted([r for r in summary if r['feature_id']==f and r['method']=='multiplicative' and r['prompt_group']==g and r['subset']==subset and r['metric']==metric and r['mean'] is not None],key=lambda r:float(r['strength_label'][6:]));xx=[float(r['strength_label'][6:]) for r in ss]
   ax.plot(xx,[r['mean'] for r in ss],color='#224488',lw=1.5);ax.fill_between(xx,[np.nan if r['ci95_low'] is None else r['ci95_low'] for r in ss],[np.nan if r['ci95_high'] is None else r['ci95_high'] for r in ss],color='#4477AA',alpha=.2)
   ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',ls=':',lw=.7);ax.axvline(1,color='green',ls=':',lw=.7);ax.axvspan(-100,-50,color='gray',alpha=.12);ax.axvspan(50,100,color='gray',alpha=.12)
   ax.set_xscale('symlog',linthresh=7);ax.set_xticks([-100,-50,-7,0,1,7,50,100]);ax.set_xticklabels(['-100','-50','-7','0','1','7','50','100'],rotation=45,fontsize=7);ax.set_title(g.replace('_',' ')+f"; n={len(set(r['seed'] for r in rs))}")
  fig.suptitle(f'Feature {f}: paired predicted beard-area change; {subset}\nFace denominator: '+('current-image parser face' if metric=='delta_current_face_pp' else 'fixed baseline parser face')+'; shade = |α|≥50 caution, not per-image distortion labels')
  fig.supxlabel('Latent multiplier α: 0 = removal; 1 = unchanged; α < 0 = sign reversal');fig.supylabel('Paired area change (percentage points)');save(fig,f'feature{f}_{gender}_{subset}_{metric}')
 # Applied norm plots retain sign via separate alpha regimes, without pretending equal norm guarantees naturalness.
 for f in features:
  fig,axes=plt.subplots(2,3,figsize=(13,7))
  for ax,g in zip(axes.flat,MEN):
   rs=[r for r in rows if int(r['feature_id'])==f and r['prompt_group']==g and r['method'] in ['multiplicative','additive'] and r['delta_current_face_pp'] is not None]
   for method,color,marker in [('multiplicative','#4477AA','o'),('sign_reversal','#AA3377','x'),('additive','#228833','^')]:
    ss=[r for r in rs if (r['method']=='additive' if method=='additive' else r['method']=='multiplicative' and (float(r['multiplier'])<0 if method=='sign_reversal' else float(r['multiplier'])>=0)) and number(r['applied_l2']) is not None];ax.scatter([float(r['applied_l2']) for r in ss],[r['delta_current_face_pp'] for r in ss],s=9,alpha=.3,c=color,marker=marker,label=method)
   ax.set_xscale('symlog',linthresh=1);ax.axhline(0,color='gray',lw=.7);ax.set_title(g.replace('_',' '));ax.legend(fontsize=7)
  fig.suptitle(f'Feature{f}: paired parser area change versus actual applied block-output L2\n20 multiplicative seeds; 5 additive seeds; equal norms do not establish specificity')
  fig.supxlabel('Applied block-output perturbation L2 (nonnegative)');fig.supylabel('Current-face normalized change (pp)');save(fig,f'feature{f}_applied_norm')
  # Fixed additive offsets share exactly the same x values; calibrated/matched offsets are shown separately versus norms.
  fig,axes=plt.subplots(2,3,figsize=(13,7))
  for ax,g in zip(axes.flat,MEN):
   rs=[r for r in rows if int(r['feature_id'])==f and r['method']=='additive' and r['prompt_group']==g and r['strength_label'] in ['fixed_10','fixed_20','fixed_50']]
   for seed in sorted({r['seed'] for r in rs}):
    ss=sorted([r for r in rs if r['seed']==seed],key=lambda r:float(r['additive_offset']));ax.plot([float(r['additive_offset']) for r in ss],[r['delta_current_face_pp'] for r in ss],color='#CC6677',lw=.8,alpha=.35)
   xx=[];yy=[];lo=[];hi=[]
   for off in [10,20,50]:
    vals=[r['delta_current_face_pp'] for r in rs if float(r['additive_offset'])==off and r['delta_current_face_pp'] is not None]
    if vals:
     xx.append(off);yy.append(np.mean(vals));ll,hh=interval(vals);lo.append(ll);hi.append(hh)
   ax.plot(xx,yy,'o-',color='#992244');ax.fill_between(xx,lo,hi,color='#CC6677',alpha=.2);ax.axhline(0,color='gray',lw=.7);ax.set_title(g.replace('_',' '));ax.set_xticks([10,20,50])
  fig.suptitle(f'Feature{f}: existing fixed-offset additive interventions, five paired seeds\nz′ = z + β; scores are parser predictions, not verified beard changes');fig.supxlabel('Additive offset β');fig.supylabel('Current-face normalized change (pp)');save(fig,f'feature{f}_additive_fixed')
 # Ten-feature overlays use the same 20 seeds per feature, no random-control claim.
 if len(features)>1:
  for method in ['multiplicative','additive']:
   fig,axes=plt.subplots(2,3,figsize=(13,7))
   for ax,g in zip(axes.flat,MEN):
    for f in features:
     ss=[r for r in summary if r['feature_id']==f and r['method']==method and r['prompt_group']==g and r['subset']=='all' and r['metric']=='delta_current_face_pp' and r['mean'] is not None]
     if method=='multiplicative':ss.sort(key=lambda r:float(r['strength_label'][6:]));x=[float(r['strength_label'][6:]) for r in ss]
     else:ss=[r for r in ss if r['strength_label'] in ['fixed_10','fixed_20','fixed_50']];ss.sort(key=lambda r:int(r['strength_label'].split('_')[1]));x=[int(r['strength_label'].split('_')[1]) for r in ss]
     ax.plot(x,[r['mean'] for r in ss],lw=1,label=str(f))
    if method=='multiplicative':ax.set_xscale('symlog',linthresh=7);ax.axvspan(-100,-50,color='gray',alpha=.12);ax.axvspan(50,100,color='gray',alpha=.12)
    ax.axhline(0,color='gray',lw=.7);ax.set_title(g.replace('_',' '))
   axes.flat[0].legend(ncol=2,fontsize=6);fig.suptitle('Ten selected features: mean paired parser-area change (not random controls)');fig.supxlabel('Latent multiplier α' if method=='multiplicative' else 'Additive offset β');fig.supylabel('Current-face normalized change (pp)');save(fig,'ten_feature_'+method)
 # Small deterministic galleries: seed30, unchanged originals beside predicted masks.
 for f in features:
  for g in MEN:
   ss=[r for r in rows if int(r['feature_id'])==f and r['prompt_group']==g and r['seed']=='30' and ((r['method']=='multiplicative' and float(r['multiplier']) in [-7,0,1,7]) or (r['method']=='additive' and r['strength_label'] in ['fixed_10','fixed_20','fixed_50']))]
   ss.sort(key=lambda r:(r['method'],number(r['multiplier']) if r['method']=='multiplicative' else number(r['additive_offset'])))
   sheet=Image.new('RGB',(len(ss)*180,390),'white');d=ImageDraw.Draw(sheet)
   for j,r in enumerate(ss):
    with Image.open(run/r['image_path']) as im:rgb=np.asarray(im.convert('RGB')).copy()
    with Image.open(run/r['beard_mask_path']) as im:mask=np.asarray(im)>0
    overlay=rgb.copy();overlay[mask]=(.45*rgb[mask]+.55*np.array([255,40,180])).astype(np.uint8)
    d.text((j*180+3,3),r['strength_label'],fill='black')
    for k,arr in enumerate([rgb,overlay]):sheet.paste(Image.fromarray(arr).resize((180,180)),(j*180,25+185*k))
   sheet.save(run/'figures'/f'gallery_feature{f}_{g}.png')
 info=dict(n_conditions=len(rows),n_unique_assets=len(cache),n_new_generated_images=0,n_trained_models=0,features=features,n_private_review_joins=len(private),bootstrap_seed=4228,n_bootstrap=10000,random_control_status='No matched-norm random controls found in inspected intervention archives; other nine selected features are not random controls')
 (run/'results/complete.json').write_text(json.dumps(info,indent=2))
 text='# Existing-image beard-parser intervention scoring\n\n'+json.dumps(info,indent=2)+'\n\n'
 text+='## Definitions and limits\n\nThe evaluator is the existing FaceParsing decoder and DINOv3 backbone. Raw class14 is listed as Beard in its README; a conflicting visualization class list remains documented. These are unvalidated parser predictions. The original helper, weights and preprocessing are unchanged.\n\nPrimary area fraction is beard / parser classes1–11 plus14 (facial components and beard; excludes neck12 and hair13). Secondary metrics divide by the same-seed baseline face area held fixed, and by full512×512 image area. Current-face normalization can itself change under edits; fixed-denominator sensitivity is provided. Zero denominators stay missing. Fixed-baseline fractions can exceed100%; they are not clipped.\n\nEvery edit is paired by feature, prompt and seed with its own control. Multiplicative controls use α=1; additive uses the saved no-intervention image. config/control_equivalence.csv records actual RGB equivalence. α=0 removes the coordinate, α=1 leaves it unchanged, α<0 reverses sign, α>1 amplifies. There are no fractional α between0 and1: no graded attenuation claim is supported within that interval. Applied L2 is recorded separately; negative multipliers are not relabeled attenuation.\n\nFull20-seed multiplicative curves and five-seed additive curves have different precision. Use results/common_seed_comparison.csv for shared seeds30–34. Per-context paired curves have one line per seed; bootstrap resamples seeds, averaging contexts within a seed before pooled summaries. Confidence intervals do not account for parser error. No multiple-testing-adjusted significance or demographic specificity is claimed. Prompt groups are conditioning text.\n\nBaseline-bearded panels use a provisional baseline beard/face fraction≥1%; thresholds0,0.5,1,2,5% are retained for sensitivity. This is an operational parser criterion, not validated beard presence. All zero-baseline contexts remain in the full table and initiation analysis. Additive initiation is a threshold crossing, not proof of new facial hair.\n\nShading |α|≥50 is a caution range motivated by an existing stratified100-case human review; it is not a universal distortion threshold. MediaPipe detection is not structural validity. Per-case review linkage, where available, stays local/private; unreviewed images are not presumed usable.\n\n## Figures\n\n'
 for f in features:
  for name in [f'feature{f}_men_all_delta_current_face_pp',f'feature{f}_men_baseline_bearded_delta_current_face_pp',f'feature{f}_additive_fixed',f'feature{f}_applied_norm']:
   text+=f'![{name}](figures/{name}.png)\n\n'
 if len(features)>1:text+='![Ten multiplicative features](figures/ten_feature_multiplicative.png)\n\n![Ten additive features](figures/ten_feature_additive.png)\n\n'
 text+='Women parser-comparison panels, fixed-baseline denominator panels, all PNG/PDF figures and seed30 paired galleries are in figures/. All arrays, original PNGs and masks remain on the research host. Existing exact-image predictions retain missing probability fields explicitly; fresh predictions preserve full class maps and lossless float32 beard-probability maps. No source images were regenerated, resized, or overwritten. Contact sheets alone use resized previews.\n\n## Reproduce and resume\n\n```bash\n'
 text+=f'# Original smoke test: feature4228,12 conditions; expanded runs reuse its verified result.\n# Shards: 0 through '+str(cfg['n_shards']-1)+f'\nsbatch config/code/score.sbatch {run} --shard 0\nPYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python {run}/config/code/analyze.py --run {run}\n```\n\nCompleted assets are SHA-verified and skipped on resume. Each shard has an exclusive lock and append-only journal. Partial attempts are retained. Missing predictions block the final analysis rather than silently disappear. Input hashes, source CSV snapshots, checkpoint hashes, runtime versions and code are in config/. Full archive: `'+str(run)+'`.\n'
 # Direct comparison table retains common seeds rather than mixing N=5 and N=20.
 write(run/'results/common_seed_comparison.csv',[r for r in rows if 30<=int(r['seed'])<=34])
 (run/'README.md').write_text(text)
 files=[p for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.csv' and p.suffix!='.lock' and '__pycache__' not in p.parts]
 def manifest_row(p):return dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p))
 with cf.ThreadPoolExecutor(max_workers=16) as pool:write(run/'output_manifest.csv',list(pool.map(manifest_row,files)))
 print('Complete analysis',json.dumps(info),flush=True)
if __name__=='__main__':main()
