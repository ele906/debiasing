"""Bounded, CPU-only analysis of existing feature-4228 conditions. No model/image generation imports."""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('/n/fs/goose')
EXPERIMENT=ROOT/'experiments/mid0_sae_selectivity/spatial4228_20261114_183629_015130'
SOURCE=EXPERIMENT/'published_report'
METHODS=('natural','uniform','shuffled','localized')
GROUPS=('black_man','east_asian_man')
OUTCOMES=('post_reencoded_discrepancy','pixel_mae','lowerface_change_fraction','outside_change_fraction','reencoding_relative_to_modified','drop_energy_fraction')
COLORS=dict(natural='#19734b',uniform='#3c50a0',shuffled='#bf7c18',localized='#ac356c')
EXPECTED='cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    if not keys:keys=['status'];rows=[dict(status='no_rows')]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def number(value):
    if value is None or str(value).strip()=='':return np.nan
    return float(value)


def mean(v,w):return float(np.sum(w*v)/np.sum(w))


def seed_bootstrap_counts(seeds):
    """126 unique multisets, weighted to enumerate all 5^5 bootstrap draws exactly."""
    n=len(seeds)
    for sequence in itertools.combinations_with_replacement(range(n),n):
        counts=np.bincount(sequence,minlength=n)
        multiplicity=math.factorial(n)//math.prod(math.factorial(int(k)) for k in counts)
        yield dict(zip(seeds,counts)),multiplicity


def design(rows,spec):
    n=len(rows);columns=[np.ones(n)];names=['intercept']
    def categorical(key):
        levels=sorted({r[key] for r in rows})
        for level in levels[1:]:columns.append(np.asarray([r[key]==level for r in rows],float));names.append(key+'='+str(level))
    if spec=='pooled_magnitude':
        categorical('target_norm')
    elif spec=='method_only':categorical('method')
    elif spec in ('adjusted','geometry_identity'):
        categorical('method');categorical('prompt_id');categorical('target_norm')
        columns.append(np.asarray([(r['actual_applied_norm']/r['target_norm']-1)*1e6 for r in rows]));names.append('applied_norm_deviation_ppm')
        columns.append(np.asarray([-.5 if r['donor_id']=='black_man_seed030' else .5 if r['donor_id']=='black_man_seed031' else 0 for r in rows]));names.append('nested_donor_contrast')
        for s in (1729,2718):
            columns.append(np.asarray([(float(r['shuffle_seed']==s)-1/3) if r['method']=='shuffled' else 0 for r in rows]));names.append('nested_shuffle_'+str(s))
        if spec=='geometry_identity':
            for k in ('pattern_support_fraction','pattern_original_overlap_fraction'):
                columns.append(np.asarray([r[k] for r in rows]));names.append(k)
    else:raise ValueError(spec)
    return np.column_stack(columns),names


def residual_fit(D,x,Y,w):
    valid=(w>0)&np.isfinite(x)&np.all(np.isfinite(Y),axis=1)&np.all(np.isfinite(D),axis=1)
    if not np.any(valid):return None
    D=D[valid];x=x[valid];Y=Y[valid];w=w[valid];sw=np.sqrt(w)
    rhs=np.column_stack((x,Y));coef,_,rank,_=np.linalg.lstsq(D*sw[:,None],rhs*sw[:,None],rcond=1e-11)
    residual=rhs-D@coef;xr=residual[:,0];yr=residual[:,1:];sxx=float(np.sum(w*xr*xr));total=float(np.sum(w*(x-mean(x,w))**2))
    fraction=sxx/total if total>1e-16 else 0.
    identifiable=sxx/max(float(w.sum()),1)<1e-12 or fraction<1e-10
    beta=None if identifiable else np.sum(w[:,None]*xr[:,None]*yr,axis=0)/sxx
    syy=np.sum(w[:,None]*yr*yr,axis=0)
    corr=None if identifiable else np.divide(np.sum(w[:,None]*xr[:,None]*yr,axis=0),np.sqrt(sxx*syy),out=np.full(Y.shape[1],np.nan),where=syy>1e-16)
    return dict(beta=beta,correlation=corr,x_residual=xr,y_residual=yr,valid=valid,sxx=sxx,total=total,residual_fraction=fraction,
                residual_sd=float(np.sqrt(sxx/w.sum())),r2=1-fraction if total>1e-16 else None,rank=int(rank),n_columns=D.shape[1],n_rows=int(valid.sum()),weights=w)


def regression(rows,spec,scope):
    D,names=design(rows,spec);x=np.asarray([r['support_expansion'] for r in rows]);w=np.asarray([r['balance_weight'] for r in rows]);seeds=sorted({r['seed'] for r in rows})
    output=[];diagnostics=[]
    for outcome in OUTCOMES:
        Y=np.asarray([[r.get(outcome,np.nan)] for r in rows]);fit=residual_fit(D,x,Y,w)
        if fit is None:output.append(dict(scope=scope,spec=spec,outcome=outcome,status='no_observed_outcome'));continue
        d=dict(scope=scope,spec=spec,outcome=outcome,n_images=fit['n_rows'],n_seed_clusters=len({r['seed'] for r,v in zip(rows,fit['valid']) if v}),
               effective_method_cells=float(w[fit['valid']].sum()),residual_support_sd=fit['residual_sd'],support_r2_controls=fit['r2'],
               residual_support_variance_fraction=fit['residual_fraction'],design_rank=fit['rank'],design_columns=fit['n_columns'],
               n_missing=len(rows)-fit['n_rows'],controls=json.dumps(names))
        diagnostics.append(d)
        if fit['beta'] is None:
            output.append(dict(d,status='support_not_identifiable_after_controls',slope_per_1pp=None,ci_low=None,ci_high=None,partial_correlation=None));continue
        beta=float(fit['beta'][0]);boot=[];invalid_mass=0
        for counts,mass in seed_bootstrap_counts(seeds):
            wb=w*np.asarray([counts[r['seed']] for r in rows]);b=residual_fit(D,x,Y,wb)
            if b is None or b['beta'] is None:invalid_mass+=mass;continue
            boot.extend([float(b['beta'][0])*.01]*mass)
        loo=[]
        for seed in seeds:
            b=residual_fit(D,x,Y,w*np.asarray([r['seed']!=seed for r in rows]))
            loo.append(None if b is None or b['beta'] is None else float(b['beta'][0])*.01)
        # Cluster score sign-flip sensitivity, NOT an exact treatment-randomization test.
        valid_rows=[r for r,v in zip(rows,fit['valid']) if v]
        scores=np.asarray([np.sum(fit['weights'][np.asarray([r['seed']==s for r in valid_rows])]*
            fit['x_residual'][np.asarray([r['seed']==s for r in valid_rows])]*fit['y_residual'][np.asarray([r['seed']==s for r in valid_rows]),0]) for s in seeds])
        signs=np.asarray(list(itertools.product((-1,1),repeat=len(seeds))));observed=abs(float(scores.sum()))
        p=float(np.mean(np.abs(signs@scores)>=observed-1e-12))
        output.append(dict(d,status='exploratory_conditional_association',slope_per_1pp=beta*.01,
            ci_low=None if not boot else float(np.quantile(boot,.025)),ci_high=None if not boot else float(np.quantile(boot,.975)),
            partial_correlation=None if not np.isfinite(fit['correlation'][0]) else float(fit['correlation'][0]),
            exact_bootstrap_total_mass=len(seeds)**len(seeds),unidentified_bootstrap_mass=invalid_mass,
            leave_one_seed_out_slopes=json.dumps(loo),cluster_score_signflip_sensitivity_p=p))
    return output,diagnostics


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path);args=parser.parse_args()
    if args.run:run=args.run.resolve();cfg=json.loads((run/'config/analysis_config.json').read_text())
    else:
        run=EXPERIMENT.parent/('support_analysis_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
        for d in ('config','inputs','results','figures','logs'):(run/d).mkdir(parents=True,exist_ok=False)
        cfg=dict(analysis_id=run.name,source=str(SOURCE),source_experiment=str(EXPERIMENT),feature=4228,layer='unet.mid_block.attentions.0',sae_sha256=EXPECTED,
            maximum_edited_rows=200,seeds=[30,31,32,33,34],weighting='Equal weight to each method in every recipient/group/target cell; average donor and shuffle repetitions',
            bootstrap='Whole recipient-seed clusters jointly resampled; enumerate all 5^5 draws through 126 multinomial count vectors',
            outcomes=list(OUTCOMES),claim='Associations of proxy metrics, not causality or visual unnaturalness',raw_tensor_audit=True)
        dump(run/'config/analysis_config.json',cfg)
        shutil.copy2(__file__,run/'analysis.py')
        input_paths=[SOURCE/'results/per_image_results.csv',SOURCE/'results/naturalness.csv',SOURCE/'configuration.csv',SOURCE/'context_activity.csv',
            EXPERIMENT/'config/evaluation_source/evaluate.py',EXPERIMENT/'config/source/worker.py',EXPERIMENT/'config/source/core.py']
        manifest=[]
        for i,path in enumerate(input_paths):
            dest=run/'inputs'/(str(i)+'_'+path.name);shutil.copy2(path,dest)
            manifest.append(dict(source=str(path),saved=str(dest.relative_to(run)),sha256=sha(path),bytes=path.stat().st_size))
        write_csv(run/'config/input_manifest.csv',manifest)
        env=sys.version+'\n'
        for cmd in (['git','rev-parse','HEAD'],['git','status','--porcelain'],[sys.executable,'-m','pip','freeze']):
            env+='\n'+repr(cmd)+'\n'+subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True).stdout
        (run/'config/environment.txt').write_text(env)
    print('ANALYSIS_DIRECTORY',run,flush=True)
    raw=list(csv.DictReader((run/'inputs/0_per_image_results.csv').open()))
    selected=[r for r in raw if r['method'] in METHODS]
    assert len(selected)==200 and len({r['image_id'] for r in selected})==200
    assert all(r['feature_id']=='4228' and r['layer']==cfg['layer'] and r['sae_sha256']==EXPECTED for r in selected)
    numeric=['target_norm','actual_applied_norm','fraction_positions_exceeding_topk','post_reencoded_discrepancy','pixel_mae','lowerface_change_fraction','outside_change_fraction',
        'pattern_nonzero_cells','introduced_support_fraction','original_activation_nonzero_fraction','original_support_mean','modified_support_mean','latent_change_l2','face_detected']
    rows=[];archives=[];mechanics=[]
    for i,s in enumerate(selected):
        r={k:s.get(k) for k in ('image_id','method','donor_id','prompt_id','group','pattern_id','activation_path','activation_sha256','generation_status')}
        for k in numeric:r[k]=number(s.get(k))
        r['seed']=int(s['seed']);r['shuffle_seed']=int(s['shuffle_seed']) if s.get('shuffle_seed') else None
        r['support_expansion']=r['fraction_positions_exceeding_topk'];r['pattern_support_fraction']=r['pattern_nonzero_cells']/256
        r['pattern_original_overlap_fraction']=r['pattern_support_fraction']-r['introduced_support_fraction']
        n=sum(t['method']==s['method'] and t['prompt_id']==s['prompt_id'] and t['target_norm']==s['target_norm'] for t in selected)
        r['balance_weight']=1/n
        archive=Path(s['activation_path']);h=sha(archive);assert h==s['activation_sha256']
        archives.append(dict(path=str(archive),sha256=h,bytes=archive.stat().st_size))
        with np.load(archive) as a:
            original=a['original_feature'].astype(np.float64);modified=a['modified_feature'].astype(np.float64);post=a['post_reencoded_feature'].astype(np.float64)
            original_full=a['original_latents'];modified_full=a['modified_latents'];pattern=a['base_pattern']
            counts=(original_full!=0).sum(-1);after=(modified_full!=0).sum(-1)
            error=(post-modified)**2;zero=post==0;total_error=float(error.sum());drop_error=float(error[zero].sum())
            d=float(np.sqrt(total_error));assert np.isclose(d,r['post_reencoded_discrepancy'],rtol=2e-6,atol=2e-6)
            sf=float(np.mean(after>10));introduced=float(np.mean((original==0)&(modified!=0)));pfrac=float(np.mean(pattern!=0));overlap=float(np.mean((pattern!=0)&(original!=0)))
            assert abs(sf-r['support_expansion'])<1e-10
            assert abs(introduced-r['introduced_support_fraction'])<1e-7
            assert abs(overlap-r['pattern_original_overlap_fraction'])<1e-7
            # A genuine full-latent k-sparsity floor, not a bound on the selected-coordinate error.
            sq=modified_full.astype(np.float64)**2
            tail=np.partition(sq,-10,axis=-1)[...,:-10]
            floor=float(np.sqrt(tail.sum()))
            r.update(pattern_original_overlap_fraction=overlap,pattern_support_fraction=pfrac,
                reencoding_relative_to_modified=d/max(float(np.linalg.norm(modified)),1e-12),
                reencoding_relative_to_intervention=d/max(r['latent_change_l2'],1e-12),
                drop_energy_fraction=drop_error/total_error if total_error>1e-20 else np.nan,
                zero_drop_error_l2=float(np.sqrt(drop_error)),retained_or_spurious_error_l2=float(np.sqrt(max(total_error-drop_error,0))),
                full_latent_best_k_sparse_error_floor=floor,modified_selected_l2=float(np.linalg.norm(modified)),
                baseline_all_positions_exactly_k=bool(np.all(counts==10)),post_zero_fraction=float(np.mean(zero)),
                support_identity_error=sf-(pfrac-overlap))
            mechanics.append({k:r[k] for k in ('image_id','method','group','seed','target_norm','support_expansion','pattern_support_fraction','pattern_original_overlap_fraction',
                'support_identity_error','baseline_all_positions_exactly_k','post_reencoded_discrepancy','zero_drop_error_l2','retained_or_spurious_error_l2',
                'drop_energy_fraction','post_zero_fraction','modified_selected_l2','reencoding_relative_to_modified','full_latent_best_k_sparse_error_floor')})
        rows.append(r)
        if (i+1)%50==0:print('Audited existing archives',i+1,flush=True)
    assert all(r['baseline_all_positions_exactly_k'] for r in rows)
    assert max(abs(r['support_identity_error']) for r in rows)<1e-10
    write_csv(run/'config/activation_input_hashes.csv',archives)
    write_csv(run/'results/per_image_analysis.csv',rows);write_csv(run/'results/mechanical_coupling.csv',mechanics)
    missing=[dict(variable=k,n_missing=sum(not np.isfinite(r[k]) for r in rows),n_total=len(rows)) for k in numeric+list(OUTCOMES)]
    write_csv(run/'results/missingness.csv',missing)
    aggregated=[]
    for group,seed,target,method in itertools.product(GROUPS,range(30,35),sorted({r['target_norm'] for r in rows}),METHODS):
        s=[r for r in rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']==method]
        out=dict(group=group,seed=seed,target_norm=target,method=method,n_images=len(s),n_independent_seed_clusters=1)
        for k in ('support_expansion',*OUTCOMES):
            v=[r[k] for r in s if np.isfinite(r[k])];out[k]=float(np.mean(v)) if v else None
            out[k+'_missing']=len(s)-len(v)
        aggregated.append(out)
    write_csv(run/'results/recipient_method_cells.csv',aggregated)
    regressions=[];diagnostics=[]
    for scope,subset in [('all_methods',rows)]+[(m,[r for r in rows if r['method']==m]) for m in METHODS]:
        specs=('pooled_magnitude','adjusted','geometry_identity') if scope=='all_methods' else ('pooled_magnitude','adjusted')
        for spec in specs:
            out,diag=regression(subset,spec,scope);regressions.extend(out);diagnostics.extend(diag)
    write_csv(run/'results/conditional_associations.csv',regressions);write_csv(run/'results/design_diagnostics.csv',diagnostics)
    summarize_design(run,rows);paired_differences(run,rows);plots(run,rows,aggregated,regressions);report(run,rows,regressions)
    manifest=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.csv']
    write_csv(run/'output_manifest.csv',manifest)
    print('DONE',run,flush=True)


def summarize_design(run,rows):
    x=np.array([r['support_expansion'] for r in rows]);w=np.array([r['balance_weight'] for r in rows]);Y=x[:,None]
    output=[]
    for spec in ('method_only','adjusted','geometry_identity'):
        D,names=design(rows,spec);f=residual_fit(D,x,Y,w)
        output.append(dict(spec=spec,weighted_r2=f['r2'],residual_support_sd=f['residual_sd'],residual_variance_fraction=f['residual_fraction'],rank=f['rank'],columns=f['n_columns']))
    write_csv(run/'results/support_explained_by_design.csv',output)
    ranges=[]
    for m in METHODS:
        subset=[r for r in rows if r['method']==m];v=np.array([r['support_expansion'] for r in subset]);ww=np.array([r['balance_weight'] for r in subset])
        ranges.append(dict(method=m,n_images=len(subset),n_seed_clusters=5,min_support=float(v.min()),max_support=float(v.max()),weighted_mean=mean(v,ww),
            weighted_sd=float(np.sqrt(mean((v-mean(v,ww))**2,ww))),n_unique_support_values=len(np.unique(v))))
    write_csv(run/'results/support_ranges.csv',ranges)
    invariance=[]
    for key in sorted({(r['prompt_id'],r['pattern_id']) for r in rows}):
        s=[r for r in rows if (r['prompt_id'],r['pattern_id'])==key]
        invariance.append(dict(prompt_id=key[0],pattern_id=key[1],n_strengths=len({r['target_norm'] for r in s}),n_unique_support=len({r['support_expansion'] for r in s})))
    write_csv(run/'results/support_strength_invariance.csv',invariance)


def paired_differences(run,rows):
    result=[];per_seed=[]
    for group,target,comparison in itertools.product(GROUPS,sorted({r['target_norm'] for r in rows}),('uniform','shuffled','localized')):
        for metric in ('support_expansion',*OUTCOMES):
            differences=[]
            for seed in range(30,35):
                ds=[]
                for donor in ('black_man_seed030','black_man_seed031'):
                    n=[r for r in rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']=='natural' and r['donor_id']==donor]
                    c=[r for r in rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']==comparison and (comparison!='shuffled' or r['donor_id']==donor)]
                    vals=[r[metric] for r in n+c]
                    if len(n)==1 and len(c)==(3 if comparison=='shuffled' else 1) and all(np.isfinite(v) for v in vals):ds.append(vals[0]-float(np.mean(vals[1:])))
                if len(ds)==2:
                    d=float(np.mean(ds));differences.append(d);per_seed.append(dict(group=group,target_norm=target,contrast='natural_minus_'+comparison,outcome=metric,seed=seed,difference=d))
            if differences:
                v=np.array(differences);ix=np.array(list(itertools.product(range(len(v)),repeat=len(v))));boot=v[ix].mean(1)
                result.append(dict(group=group,target_norm=target,contrast='natural_minus_'+comparison,outcome=metric,mean_difference=float(v.mean()),
                    ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),n_seed_clusters=len(v),n_missing_seed_clusters=5-len(v)))
            else:result.append(dict(group=group,target_norm=target,contrast='natural_minus_'+comparison,outcome=metric,mean_difference=None,ci_low=None,ci_high=None,n_seed_clusters=0,n_missing_seed_clusters=5))
    write_csv(run/'results/paired_differences.csv',result);write_csv(run/'results/paired_seed_differences.csv',per_seed)


def plots(run,rows,cells,models):
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    def save(fig,name):fig.tight_layout();fig.savefig(run/'figures'/(name+'.pdf'));fig.savefig(run/'figures'/(name+'.png'),dpi=300);plt.close(fig)
    outcomes=('post_reencoded_discrepancy','pixel_mae','lowerface_change_fraction')
    labels=('Selected-coordinate re-encoding L2','Image pixel MAE','Lower-face fraction of pixel change')
    targets=sorted({r['target_norm'] for r in rows})
    fig,axes=plt.subplots(3,4,figsize=(15,10))
    for j,(group,target) in enumerate(itertools.product(GROUPS,targets)):
        for i,(metric,label) in enumerate(zip(outcomes,labels)):
            ax=axes[i,j]
            for method in METHODS:
                s=[r for r in rows if r['group']==group and r['target_norm']==target and r['method']==method]
                ax.scatter([r['support_expansion'] for r in s],[r[metric] for r in s],s=17,alpha=.42,color=COLORS[method],label=method)
                s=[r for r in cells if r['group']==group and r['target_norm']==target and r['method']==method]
                ax.scatter([r['support_expansion'] for r in s],[r[metric] for r in s],s=34,marker='D',edgecolor='black',linewidth=.3,color=COLORS[method])
            ax.set(xlabel='Fraction of cells exceeding k=10',ylabel=label,title=f'{group}\nApplied L2 target {target}');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('Raw images (faint) and donor/shuffle-averaged recipient-method cells (diamonds); five paired seeds');save(fig,'01_support_vs_responses')
    fig,axes=plt.subplots(4,3,figsize=(13,12))
    for i,method in enumerate(METHODS):
        for j,(metric,label) in enumerate(zip(outcomes,labels)):
            ax=axes[i,j]
            for group,marker in zip(GROUPS,('o','s')):
                for target,size in zip(targets,(24,55)):
                    s=[r for r in rows if r['method']==method and r['group']==group and r['target_norm']==target]
                    ax.scatter([r['support_expansion'] for r in s],[r[metric] for r in s],s=size,marker=marker,alpha=.5,
                        facecolors=COLORS[method] if target==targets[1] else 'none',edgecolors=COLORS[method],label=group+' / '+str(target))
            ax.set(title=method,xlabel='Support expansion (within-method scale)',ylabel=label);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=6);fig.suptitle('Within-method associations; group = marker, target = marker size/fill; repeats are not independent');save(fig,'02_within_method')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,m in enumerate(METHODS):
        s=[r for r in rows if r['method']==m];axes[0].scatter([r['support_expansion'] for r in s],[i]*len(s),s=20,alpha=.3,color=COLORS[m])
    axes[0].set(yticks=range(4),yticklabels=METHODS,xlabel='Support expansion',title='Method-specific support ranges')
    diagnostics=list(csv.DictReader((run/'results/support_explained_by_design.csv').open()))
    axes[1].bar(range(3),[float(r['weighted_r2']) for r in diagnostics],color=['#697ca5','#46686a','#738b55'])
    axes[1].set(xticks=range(3),xticklabels=['Method only','Method + paired design','Pattern support − overlap'],ylabel='Explained support variance (weighted R²)',ylim=(0,1.05))
    save(fig,'03_identifiability')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    D,_=design(rows,'adjusted');x=np.array([r['support_expansion'] for r in rows]);w=np.array([r['balance_weight'] for r in rows])
    for ax,metric,label in zip(axes,outcomes,labels):
        f=residual_fit(D,x,np.asarray([[r[metric]] for r in rows]),w)
        for m in METHODS:
            mask=np.array([r['method']==m for r in rows])[f['valid']]
            ax.scatter(100*f['x_residual'][mask],f['y_residual'][mask,0],s=20,alpha=.4,color=COLORS[m],label=m)
        ax.set(xlabel='Residual support (percentage points)',ylabel='Residual '+label,title='After method, recipient, target, donor/shuffle controls');ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
    axes[0].legend(fontsize=6);save(fig,'04_conditional_residuals')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for m in METHODS:
        s=[r for r in rows if r['method']==m]
        axes[0].scatter([r['support_expansion'] for r in s],[r['drop_energy_fraction'] for r in s],s=18,alpha=.4,color=COLORS[m],label=m)
        axes[1].scatter([r['post_reencoded_discrepancy']**2 for r in s],[r['zero_drop_error_l2']**2 for r in s],s=18,alpha=.4,color=COLORS[m])
        axes[2].scatter([r['support_expansion'] for r in s],[r['full_latent_best_k_sparse_error_floor'] for r in s],s=18,alpha=.4,color=COLORS[m])
    axes[0].set(xlabel='Support expansion',ylabel='Fraction of selected error energy from post=0 cells',title='Exact selected-error decomposition');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Total selected-coordinate error squared',ylabel='Post-zero error energy',title='Mechanical component, not semantic instability')
    axes[2].set(xlabel='Support expansion',ylabel='Best full-latent k-sparse projection L2 floor',title='Not a lower bound on selected error')
    save(fig,'05_mechanical_coupling')


def report(run,rows,models):
    design_rows=list(csv.DictReader((run/'results/support_explained_by_design.csv').open()))
    r2=float(design_rows[0]['weighted_r2']);fullr2=float(design_rows[1]['weighted_r2'])
    def fmt(v):return 'not identified' if v is None or str(v)=='' else f'{float(v):.5g}'
    lines=['# Support expansion and intervention-response proxies: bounded analysis','',
        '## Scope and definitions','',
        'Exactly 200 existing edited conditions for feature 4228, face-fine-tuned SAE SHA-256 '+EXPECTED+', SDXL Turbo, mid.0. No images were generated, no model was loaded or trained, and no annotations were requested. Ten reused baselines and ten no-op replays are excluded from the analysis sample.','',
        '- Support expansion: fraction of 256 positions whose full modified latent vector has more than k=10 nonzero coordinates. This is not the fraction of nonzero entries in the selected feature alone.',
        '- Post-reencoding discrepancy: L2 norm of the re-encoded **selected feature 4228** minus its edited coordinate map. It is not full-latent error or decoded reconstruction MSE.',
        '- Pixel MAE: mean over RGB channels and all 512×512 pixels of absolute edited-minus-baseline difference, on the 0–255 scale. Image change is not automatically instability or degradation.',
        '- Lower-face concentration: sum of per-pixel RGB MAE times a frozen soft baseline lower-face mask divided by total per-pixel MAE. Outside concentration is one minus that fraction. This is a geometric ROI measure, not beard coverage or an image-wide diffusion-distance metric.','',
        '## Design identifiability','',
        f'Method alone explains **{100*r2:.3f}%** of support variance under equal method/recipient/strength weighting. Method, paired recipient, strength, applied-norm jitter and nested donor/shuffle controls explain **{100*fullr2:.3f}%**.',
        'Support is identical across both strengths for every fixed recipient/pattern pair. Uniform support values do not overlap the natural or shuffled ranges. Thus a cross-method coefficient largely compares different intervention designs, not independently manipulated support.',
        'All original latent vectors in these 200 archives have exactly 10 nonzero coordinates. Only coordinate 4228 is edited, with nonnegative addition. Consequently, support expansion equals pattern nonzero fraction minus overlap with positions where the original selected feature was already active. This identity holds exactly in all 200 cases. Conditioning on these geometry components removes all independent support variation.','',
        '## Conditional effect sizes','',
        'Slopes below are changes per **one percentage point** of support expansion. They are observational associations, not out-of-sample predictive performance or causal effects. Intervals jointly resample the five recipient seeds, retaining prompt groups, strengths, donors and shuffled repetitions.','',
        '| Scope | Outcome | Controls | Slope / 1 pp | 95% seed-bootstrap interval | Seed clusters |','|---|---|---|---:|---|---:|']
    for r in models:
        if r['spec']=='adjusted' and r['outcome'] in ('post_reencoded_discrepancy','pixel_mae','lowerface_change_fraction'):
            lines.append(f"| {r['scope']} | {r['outcome']} | Paired design + method when applicable | {fmt(r.get('slope_per_1pp'))} | [{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}] | {r.get('n_seed_clusters',0)} |")
    lines +=['','## Mechanical coupling','',
        'The re-encoder itself reapplies nonnegative Top-K. A modified vector with 11 nonzero coordinates cannot be represented unchanged by an encoder limited to 10. That creates a nonzero **full-latent** best-k-sparse approximation floor, computed from the squared mass outside the largest ten coordinates. However, the re-encoder may drop another coordinate rather than feature 4228, so this is not a lower bound on the reported selected-coordinate discrepancy.',
        'For the selected feature, the error decomposes exactly into squared edited activation on cells where re-encoding sets that coordinate to zero, plus squared discrepancies where it remains positive. The first component is partly a direct consequence of coordinate competition and intervention scale. Its empirical share is recorded per image; it is not new evidence of semantic or visual instability.',
        'The support identity and projection floor are mathematical audits. They should not be marketed as discoveries about image naturalness. Full post-reencoded latent vectors were not saved; no full-latent actual re-encoding error is invented.','',
        '## Inference and limitations','',
        'Each recipient/group/target/method cell receives total weight one: natural donor repeats receive 1/2 each, shuffled donor/permutation repeats 1/6 each, uniform and localized conditions one each. This balances methods instead of letting the 120 shuffled images dominate. All inference uses five paired recipient-seed clusters, not 200 independent samples. Donor maps and shuffle seeds are fixed; no generalization to other donors/features is claimed.',
        'Conditional models include method indicators, recipient (prompt group × seed) fixed effects, target indicators, actual applied-L2 deviations within target in ppm, a centered donor contrast, and centered shuffled-seed contrasts. Uniform and localized support are constant within recipient, so support is absorbed by recipient effects in those within-method models.',
        'Exact empirical seed-bootstrap intervals enumerate all 5^5 resamples through 126 multinomial count vectors. Unidentified bootstrap samples are counted and disclosed; intervals use identifiable samples only when necessary. Leave-one-seed-out coefficients expose sensitivity. Five-cluster score sign-flip values are sensitivity checks under cluster-symmetry assumptions, not treatment-randomization p-values. There is no multiplicity-adjusted confirmatory claim.',
        'Missing outcomes remain missing. Case counts and missingness are exported. All face detections succeeded in the original experiment, so this sample has no nonconstant face-failure outcome with which to establish operational instability.','',
        '## Explicit research answers','',
        '1. **Does support expansion predict re-encoding inconsistency?** See pooled and adjusted effect estimates in conditional_associations.csv. A pooled association is plausible but partly mechanically coupled to Top-K competition and selected-coordinate amplitude. It is not independent evidence of semantic instability; adjusted estimates must be interpreted within the small residual design variation.',
        '2. **Does it predict spatially widespread image changes?** Outside-ROI concentration and pixel MAE are analyzed separately. Their associations cannot establish visual unnaturalness; method-specific and adjusted estimates, paired contrasts and leave-one-seed-out sensitivity determine how much of a pooled association remains.',
        '3. **Do relationships remain after controlling for method?** Statistical adjusted slopes are reported rather than assumed. Support is not identifiable within uniform/localized layouts after recipient controls, and becomes exactly nonidentifiable when pattern support and original overlap are included. Any remaining natural/shuffled association reflects limited donor/overlap geometry and five seeds, not independent support manipulation.',
        '4. **Meaningful finding or intervention-design consequence?** The defensible finding is a design/identifiability audit: matched norms do not equalize support, method explains nearly all its variation, and a re-encoding metric partly measures mechanics it is structurally predisposed to detect. This dataset cannot identify support expansion as an independent causal driver of intervention instability.',
        '5. **Smallest additional experiment?** Within one fixed layout and a few paired inactive recipients, compare unrestricted addition with a feature-retaining support-restoration policy (drop one specified pre-existing non-target coordinate at each newly active position), plus a deletion-only sham. Reuse the unrestricted images, preflight feasibility and match actual total perturbation norms. This is a small three-arm policy test, not a new broad sweep. Record amplitude and coordinate changes: deleting other coordinates is a confound, so it identifies the effect of that specific restoration policy, not support count in isolation. A subsequent within-layout randomized support-density × L2 factorial would test robustness if warranted. No such experiment is run here.','',
        '## Files and reproduction','',
        'Run `PYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python analysis.py --run ANALYSIS_DIRECTORY` to reproduce tables and figures from preserved inputs and existing hash-verified activation archives. The program imports no diffusion, SAE or image-generation modules. Inputs are copied with hashes; archive inputs are referenced and hashed. Code, versions, configuration, tables and output hashes are retained.','']
    (run/'README.md').write_text('\n'.join(lines))
    print('DESIGN_R2',r2,fullr2,flush=True)
    for r in models:
        if r['scope']=='all_methods' and r['spec']=='adjusted' and r['outcome'] in OUTCOMES:
            print('ADJUSTED',r['outcome'],r.get('slope_per_1pp'),r.get('ci_low'),r.get('ci_high'),flush=True)

if __name__=='__main__':main()
