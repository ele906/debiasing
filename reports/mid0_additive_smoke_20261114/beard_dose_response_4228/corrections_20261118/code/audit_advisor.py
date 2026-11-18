"""Audit and correct additive x-axis metadata; all image and beard measurements unchanged."""
import argparse,csv,json,collections,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import BASE,read,write,sha
MEN=['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man']
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True)
 rawpath=BASE/'ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv';raw=read(rawpath);lookup={'add_'+r['image_id']:r for r in raw};assert len(raw)==4200
 differences=[];inputs=[rawpath]
 for name,old in [('feature4228',BASE/'beard_interventions_20261117_020843_219166'),('all_features',BASE/'beard_all_features_20261117_021807_655293')]:
  dest=out/name
  if dest.exists():raise FileExistsError(dest)
  for d in ['results','config','figures']:(dest/d).mkdir(parents=True,exist_ok=True)
  shutil.copy2(old/'README.md',dest/'README.md')
  source=old/'results/per_image_scores.csv';inputs.append(source);rows=read(source);n=0
  for r in rows:
   if r['method']=='additive':
    original=lookup[r['condition_id']];previous=float(r['applied_l2']);corrected=float(original['decoded_perturbation_norm']);assert corrected>0 and float(original['perturbation_norm'])==previous
    r['applied_l2']=original['decoded_perturbation_norm'];r['latent_change_l2']=original['perturbation_norm'];r['applied_l2_source']='decoded_perturbation_norm';n+=1
    if name=='all_features':differences.append(dict(condition_id=r['condition_id'],feature_id=r['feature_id'],previous_latent_l2=previous,corrected_applied_l2=corrected,relative_difference=abs(previous-corrected)/corrected))
   else:r['applied_l2_source']='control_zero' if r['method']=='baseline' else 'delta_fro_applied'
  assert n==(420 if name=='feature4228' else 4200)
  write(dest/'results/per_image_scores.csv',rows);write(dest/'results/common_seed_comparison.csv',[r for r in rows if 30<=int(r['seed'])<=34])
  for filename in ['paired_summary.csv','baseline_threshold_sensitivity.csv','additive_initiation.csv','additive_initiation_summary.csv','metric_availability.csv','complete.json']:
   if (old/'results'/filename).exists():shutil.copy2(old/'results'/filename,dest/'results'/filename)
  means=collections.defaultdict(list)
  for r in rows:
   if 30<=int(r['seed'])<=34 and r['applied_l2']:means[r['feature_id'],r['method'],r['prompt_group'],r['strength_label']].append(float(r['applied_l2']))
  common=read(old/'results/common_seed_summary.csv')
  for r in common:r['mean_applied_l2']=float(np.mean(means[r['feature_id'],r['method'],r['prompt_group'],r['strength_label']]))
  write(dest/'results/common_seed_summary.csv',common)
  conditions=read(old/'config/conditions.csv')
  for r in conditions:
   if r['method']=='additive':r['applied_l2']=lookup[r['condition_id']]['decoded_perturbation_norm']
  write(dest/'config/conditions.csv',conditions)
  (dest/'config/original_source_run.json').write_text(json.dumps(dict(original_run=str(old),correction='Only additive applied_l2 fields updated from recorded decoded_perturbation_norm. All original archives and all y values/paired CIs remain unchanged.'),indent=2))
  if name=='all_features':
   plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
   for f in sorted({r['feature_id'] for r in rows},key=int):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,g in zip(axes.flat,MEN):
     rr=[r for r in rows if r['feature_id']==f and r['prompt_group']==g and r['delta_current_face_pp'] and r['method'] in ['multiplicative','additive']]
     for label,color,marker,selector in [('Multiplier α ≥ 0','#4477AA','o',lambda r:r['method']=='multiplicative' and float(r['multiplier'])>=0),('Multiplier α < 0','#AA3377','x',lambda r:r['method']=='multiplicative' and float(r['multiplier'])<0),('Additive','#228833','^',lambda r:r['method']=='additive')]:
      rr2=[r for r in rr if selector(r)];ax.scatter([float(r['applied_l2']) for r in rr2],[float(r['delta_current_face_pp']) for r in rr2],c=color,marker=marker,s=10,alpha=.35,label=label)
     ax.set_xscale('symlog',linthresh=1);ax.axhline(0,color='gray',lw=.7);ax.set_title(g.replace('_',' ').title());ax.legend(fontsize=7,frameon=False)
    fig.suptitle(f'Feature {f}: beard area change vs intervention size',fontsize=16);fig.supxlabel('Applied intervention size (L2)');fig.supylabel('Area change (%)')
    for ext in ['png','pdf']:fig.savefig(dest/'figures'/f'feature{f}_applied_norm.{ext}',dpi=220)
    plt.close(fig)
 write(out/'norm_field_corrections.csv',differences)
 first=read(out/'feature4228/results/paired_summary.csv');counts=[dict(prompt_group=r['prompt_group'],multiplier=r['strength_label'],n_valid_normalized_scores=r['n_seed_clusters'],n_evaluated=20) for r in first if r['method']=='multiplicative' and r['subset']=='all' and r['metric']=='delta_current_face_pp' and r['prompt_group']!='all_contexts'];write(out/'per_point_seed_counts.csv',counts)
 report=dict(n_additive_edits=4200,n_saved_pixel_identical=sum(r['pixels_identical']=='True' for r in raw),saved_pixel_mae_min=min(float(r['pixel_mae']) for r in raw),saved_pixel_mae_max=max(float(r['pixel_mae']) for r in raw),n_pixel_equal_baseline_checks=600,additive_grid_shape=[16,16],addition_applies_to_all_positions=True,overlay='modified SAE coordinate, not attention; baseline uses original coordinate',max_relative_l2_field_correction=max(r['relative_difference'] for r in differences),nominal_multiplicative_seeds=list(range(30,50)),additive_seeds=list(range(30,35)),gallery_seed=30,bootstrap_resamples=10000,y_values_changed=False,images_changed=False,models_run=0)
 (out/'audit_summary.json').write_text(json.dumps(report,indent=2));write(out/'input_hashes.csv',[dict(path=str(p),sha256=sha(p)) for p in inputs]);shutil.copy2(__file__,out/'audit_advisor.py');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
