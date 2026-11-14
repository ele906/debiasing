"""Paired seed-clustered analyses and explicitly pending blinded semantic measurements."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity');sys.path.insert(0,str(BASE))
from run import atomic_json,csv_write,inventory,digest
from collect import collect

METHODS=('natural','uniform','shuffled','localized')
COLORS=dict(natural='#176b48',uniform='#394c9b',shuffled='#ad6c16',localized='#a32c63')


def interval(values):
    v=np.asarray(values,float)
    if not len(v):return None,None,None,None
    # All n^n seed-cluster bootstrap resamples for n<=5: exact empirical bootstrap, no Monte Carlo error.
    choices=np.asarray(list(itertools.product(range(len(v)),repeat=len(v))))
    boot=v[choices].mean(1)
    signs=np.asarray(list(itertools.product((-1,1),repeat=len(v))))
    observed=abs(float(v.mean()));p=float(np.mean(np.abs((signs*v).mean(1))>=observed-1e-12))
    return float(v.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),p


def review_sheets(run,rows):
    folder=run/'evaluations/blinded_review';folder.mkdir(exist_ok=True)
    if (folder/'review_form.csv').exists():return
    baseline={r['prompt_id']:r for r in rows if r['method']=='baseline' and r['generation_status']=='complete'}
    edits=[r for r in rows if r['method']!='baseline' and r['generation_status']=='complete']
    rng=np.random.default_rng(20261114);rng.shuffle(edits);form=[];key=[]
    for page_start in range(0,len(edits),8):
        page=page_start//8+1;sheet=Image.new('RGB',(1024,4*300),'white');draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(edits[page_start:page_start+8]):
            pair_id=f'P{page_start+k+1:04d}';swap=bool(rng.integers(2));base=baseline[r['prompt_id']]
            sources=[r,base] if swap else [base,r]
            col=k%2;line=k//2;x=col*512;y=line*300
            draw.text((x+5,y+4),pair_id+'     Left                       Right',fill='black')
            for j,s in enumerate(sources):
                with Image.open(s['image_path']) as im:tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(x+j*256,y+30))
            key.append(dict(pair_id=pair_id,image_id=r['image_id'],left_is_edit=swap,baseline_image_id=base['image_id']))
            form.append(dict(pair_id=pair_id,page=f'page_{page:02d}.png',reviewer_id='',left_facial_hair_present='',right_facial_hair_present='',
                left_full_beard='',right_full_beard='',left_facial_hair_coverage_pct='',right_facial_hair_coverage_pct='',
                left_structurally_valid='',right_structurally_valid='',left_artifact_severity='',right_artifact_severity='',notes=''))
        sheet.save(folder/f'page_{page:02d}.png')
    csv_write(folder/'review_form.csv',form);csv_write(run/'evaluations/blinding_key.csv',key)
    (folder/'INSTRUCTIONS.md').write_text('Review pages without opening the blinding key or condition report. Enter reviewer_id and independent left/right labels. Facial hair present and full beard: 0=no, 1=yes, blank=uncertain/unscorable. Coverage: estimated percentage of visible lower-face skin covered by hair, 0–100, blank if uncertain. This is a human estimate, not a segmentation measurement. Structural validity: 0=invalid, 1=valid, blank=uncertain. Artifact severity: 0=none, 1=mild, 2=moderate, 3=severe. Include moustache, goatee and stubble in any-facial-hair presence; full beard is a separate judgment. Rate images independently, not only visible pair differences. Save additional raters as ratings_REVIEWER.csv in this folder. No-op pairs are included for consistency checks; their identities are withheld.\n')


def human_labels(run):
    keyfile=run/'evaluations/blinding_key.csv'
    keys={r['pair_id']:r for r in csv.DictReader(keyfile.open())};observations=[];errors=[]
    folder=run/'evaluations/blinded_review'
    for file in [folder/'review_form.csv',*sorted(folder.glob('ratings_*.csv'))]:
        for r in csv.DictReader(file.open()):
            if r['pair_id'] not in keys:errors.append(dict(file=str(file),pair_id=r['pair_id'],error='unknown pair'));continue
            k=keys[r['pair_id']];sign=1 if k['left_is_edit']=='True' else -1
            for metric,upper in [('facial_hair_present',1),('full_beard',1),('facial_hair_coverage_pct',100),('structurally_valid',1),('artifact_severity',3)]:
                l=r.get('left_'+metric,'');rr=r.get('right_'+metric,'')
                if l=='' or rr=='':continue
                try:
                    l=float(l);rr=float(rr)
                    if not(0<=l<=upper and 0<=rr<=upper):raise ValueError('out of range')
                    observations.append(dict(image_id=k['image_id'],pair_id=r['pair_id'],reviewer_id=r.get('reviewer_id',''),metric=metric,
                        baseline=l if sign==-1 else rr,edited=rr if sign==-1 else l,change=sign*(l-rr),source=str(file)))
                except ValueError as exc:errors.append(dict(file=str(file),pair_id=r['pair_id'],error=str(exc)))
    csv_write(run/'evaluations/human_labels.csv',observations)
    if not observations:csv_write(run/'evaluations/human_labels.csv',[dict(status='pending_blinded_review',n_ratings=0)])
    csv_write(run/'evaluations/rating_errors.csv',errors or [dict(status='no_errors')])
    grouped={}
    for o in observations:grouped.setdefault((o['image_id'],o['metric']),[]).append(o['change'])
    return {k:float(np.mean(v)) for k,v in grouped.items()},len(observations)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    cfg=json.loads((run/'config/experiment_config.json').read_text());rows=collect(run)
    metrics=json.loads((run/'evaluations/image_metrics.json').read_text())
    face={r['image_id']:r for r in csv.DictReader((run/'evaluations/face_validity.csv').open())}
    for r in metrics:
        e=face.get(r['image_id'],{});r['face_detected']=None if e.get('status')!='complete' else float(e['face_detected']=='True')
    review_sheets(run,rows);labels,nratings=human_labels(run)
    for r in metrics:
        for name in ('facial_hair_present','full_beard','facial_hair_coverage_pct','structurally_valid','artifact_severity'):
            r['human_'+name+'_change']=labels.get((r['image_id'],name))
    edits=[r for r in metrics if r['method'] in METHODS and r.get('match_passed')]
    groups=['black_man','east_asian_man'];targets=cfg['targets'];donors=cfg['donor_ids']
    features=['pixel_mae','ssim','lowerface_mae','outside_lowerface_mae','lowerface_change_fraction','demographic_tv','face_detected',
        'human_facial_hair_present_change','human_full_beard_change','human_facial_hair_coverage_pct_change','human_structurally_valid_change']
    def cell(group,seed,target,method,donor=None):
        return [r for r in edits if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']==method and (donor is None or r['donor_id']==donor)]
    summaries=[]
    for group in groups:
        for target in targets:
            for method in METHODS:
                for metric in features:
                    seed_values=[]
                    for seed in range(30,35):
                        subset=cell(group,seed,target,method);v=[r.get(metric) for r in subset]
                        if v and all(x is not None for x in v):seed_values.append(float(np.mean(v)))
                    mean,lo,hi,pvalue=interval(seed_values)
                    summaries.append(dict(group=group,method=method,target_norm=target,metric=metric,mean=mean,ci_low=lo,ci_high=hi,n_seed_clusters=len(seed_values),
                        n_missing_seed_clusters=5-len(seed_values),status='observed' if seed_values else 'pending_or_unavailable'))
    csv_write(run/'results/summary_statistics.csv',summaries)
    contrasts=[]
    for group in groups:
        for target in targets:
            for comparison in ('uniform','shuffled','localized'):
                for donor_view in [*donors,'donor_average']:
                    for metric in features:
                        values=[];per_seed=[]
                        for seed in range(30,35):
                            ds=donors if donor_view=='donor_average' else [donor_view];deltas=[]
                            for donor in ds:
                                natural=cell(group,seed,target,'natural',donor)
                                reference=cell(group,seed,target,comparison,donor if comparison=='shuffled' else None)
                                expected=3 if comparison=='shuffled' else 1
                                if len(natural)!=1 or len(reference)!=expected:continue
                                v=[r.get(metric) for r in natural+reference]
                                if any(x is None for x in v):continue
                                deltas.append(v[0]-float(np.mean(v[1:])))
                            if len(deltas)==len(ds):values.append(float(np.mean(deltas)));per_seed.append(dict(seed=seed,difference=values[-1]))
                        mean,lo,hi,pvalue=interval(values)
                        contrasts.append(dict(group=group,target_norm=target,contrast='natural_minus_'+comparison,donor=donor_view,metric=metric,
                            mean_difference=mean,ci_low=lo,ci_high=hi,exact_two_sided_signflip_p=pvalue,n_seed_clusters=len(values),n_missing_seed_clusters=5-len(values),
                            per_seed=json.dumps(per_seed),status='observed_exploratory' if values else 'pending_or_unavailable'))
    csv_write(run/'results/paired_comparisons.csv',contrasts)
    csv_write(run/'results/semantic_attribute_changes.csv',[r for r in contrasts if r['metric'].startswith('human_')])
    csv_write(run/'results/per_image_results.csv',metrics)
    csv_write(run/'results/failures_and_missingness.csv',[dict(image_id=r['image_id'],method=r['method'],status=r['generation_status']) for r in rows if r['generation_status']!='complete'] or [dict(status='all_conditions_complete')])
    figdir=run/'figures'
    def save(fig,name):fig.tight_layout();fig.savefig(figdir/(name+'.pdf'));fig.savefig(figdir/(name+'.png'),dpi=300);plt.close(fig)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    # 1. Raw donor maps, displayed at their original spatial resolution.
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,donor in zip(axes,donors):
        with np.load(run/'patterns/donors'/(donor+'.npz')) as data:p=data['pattern']
        im=ax.imshow(p,cmap='magma',interpolation='nearest');ax.set(title=f'{donor}\n{np.count_nonzero(p)} nonzero cells',xlabel='16×16 x',ylabel='16×16 y');fig.colorbar(im,ax=ax,label='Original activation')
    save(fig,'01_donor_activation_heatmaps')
    fig,axes=plt.subplots(2,6,figsize=(14,5))
    for i,donor in enumerate(donors):
        specs=[('Natural',run/'patterns/donors'/(donor+'.npz')),('Uniform',run/'patterns/uniform/ones.npz')]
        specs +=[(f'Shuffle {seed}',run/'patterns/shuffled'/f'{donor}_shuffle{seed}.npz') for seed in cfg['shuffle_seeds']]
        specs +=[('Lower-face ROI',run/'masks/east_asian_man_seed030.npz')]
        for ax,(name,path) in zip(axes[i],specs):
            with np.load(path) as data:p=data['pattern']
            im=ax.imshow(p,cmap='magma',interpolation='nearest');ax.set_title(name);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,fraction=.05)
    fig.suptitle('Raw pattern values; scalar coefficients subsequently solve for matched applied norms');save(fig,'02_pattern_controls')
    baseline={r['prompt_id']:r for r in metrics if r['method']=='baseline'}
    for group in groups:
        fig,axes=plt.subplots(5,6,figsize=(14,12))
        for i,seed in enumerate(range(30,35)):
            pid=f'{group}_seed{seed:03d}'
            choices=[baseline[pid],cell(group,seed,targets[1],'uniform')[0],cell(group,seed,targets[1],'natural',donors[0])[0],
                cell(group,seed,targets[1],'natural',donors[1])[0],next(r for r in cell(group,seed,targets[1],'shuffled',donors[0]) if r['shuffle_seed']==1729),
                cell(group,seed,targets[1],'localized')[0]]
            for ax,r,title in zip(axes[i],choices,['Baseline','Uniform','Natural donor 30','Natural donor 31','Shuffle donor30 /1729','Localized']):
                with Image.open(r['image_path']) as im:ax.imshow(im)
                ax.axis('off');ax.set_title(f'Seed {seed}: {title}',fontsize=8)
        fig.suptitle(group+f'; target applied L2={targets[1]}; shuffle 1729 selected in advance');save(fig,'03_paired_'+group)
    # 4/5/6. Semantic ratings are shown only when actually supplied.
    for name,metric,ylabel in [('04_facial_hair_vs_norm','human_facial_hair_present_change','Human-rated change in facial-hair presence'),
            ('05_face_detection_vs_norm','face_detected','MediaPipe detection rate; not structural validity'),
            ('06_spatial_concentration','lowerface_change_fraction','Fraction of absolute pixel change inside frozen lower-face ROI')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,group in zip(axes,groups):
            found=False
            for method in METHODS:
                s=[r for r in summaries if r['group']==group and r['method']==method and r['metric']==metric and r['mean'] is not None]
                if not s:continue
                found=True;x=np.asarray([r['target_norm'] for r in s]);y=np.asarray([r['mean'] for r in s]);lo=np.asarray([r['ci_low'] for r in s]);hi=np.asarray([r['ci_high'] for r in s])
                ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],marker='o',capsize=3,color=COLORS[method],label=method)
            if not found:ax.text(.5,.5,'Awaiting blinded facial-hair ratings\nNo semantic scores have been invented',transform=ax.transAxes,ha='center',va='center')
            else:ax.legend(fontsize=8)
            ax.set(title=group,xlabel='Actual applied L2 target',ylabel=ylabel)
        fig.suptitle('95% empirical seed-cluster bootstrap; n=5, exploratory');save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    contexts=[f'{g}_seed{s:03d}' for g in groups for s in range(30,35)]
    for ax,target in zip(axes,targets):
        array=np.full((2,10),np.nan)
        for i,donor in enumerate(donors):
            for j,pid in enumerate(contexts):
                n=[r for r in edits if r['prompt_id']==pid and r['donor_id']==donor and r['method']=='natural' and r['target_norm']==target]
                s=[r for r in edits if r['prompt_id']==pid and r['donor_id']==donor and r['method']=='shuffled' and r['target_norm']==target]
                if n and len(s)==3:array[i,j]=n[0]['lowerface_change_fraction']-np.mean([r['lowerface_change_fraction'] for r in s])
        vmax=max(float(np.nanmax(np.abs(array))),1e-6);im=ax.imshow(array,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto');fig.colorbar(im,ax=ax,label='Natural − shuffle: lower-face change fraction')
        ax.set(yticks=[0,1],yticklabels=donors,xticks=range(10),xticklabels=[p.replace('_man_seed','/').replace('east_asian','EA') for p in contexts],title=f'Target L2={target}')
        ax.tick_params(axis='x',rotation=65)
    save(fig,'07_per_donor_recipient')
    csv_write(run/'results/naturalness.csv',[{k:v for k,v in r.items() if k in ('image_id','method','group','seed','donor_id','target_norm','actual_applied_norm','coefficient','pattern_nonzero_cells','pattern_energy_in_lowerface','change_effective_positions','max_latent_change','modified_support_mean','fraction_positions_exceeding_topk','introduced_support_fraction','post_reencoded_discrepancy')} for r in edits])
    maxerr=max(r['match_relative_error'] for r in edits)
    complete=sum(r['generation_status']=='complete' for r in rows)
    info=dict(planned_conditions=len(rows),complete_conditions=complete,edited_conditions=len(edits),max_relative_norm_error=maxerr,
        human_rating_observations=nratings,face_detection_count=sum(r.get('face_detected')==1 for r in metrics),metric_rows=len(metrics))
    atomic_json(run/'results/validation_summary.json',info)
    text=f'''# Spatial-pattern experiment: exploratory results\n\n{complete}/{len(rows)} planned conditions complete; {len(edits)} edited conditions at matched targets 42.5242 and 65.1363. Maximum relative norm mismatch: {maxerr:.8g}. Two frozen donor maps, three shuffles each, and five paired seeds.\n\nAll no-op replays were checked against saved baseline pixels. Feature activation was checked per spatial map: East-Asian-man seed 33 has two active cells, while seeds 30,31,32,34 are exactly inactive. Black-man seeds 32 and 34 are also inactive.\n\n{nratings} blinded human-rating observations currently available. Facial-hair presence, coverage and structural-validity effects remain pending when corresponding ratings are absent. The facial-hair figure explicitly displays this missing measurement. Pixel differences, SSIM, face detection and demographic classifier output are not substitutes for beard judgments.\n\nPaired comparisons average donor and shuffle replicates inside each seed, then compare natural versus uniform, shuffled and localized patterns. Intervals use the exact empirical seed-cluster bootstrap (all 5^5 resamples for n=5). Two-sided sign-flip p-values enumerate all 2^5 sign assignments. All analyses are exploratory, with no multiple-comparison-adjusted confirmatory claim. Intervals can be degenerate at boundaries; five seeds provide limited population inference. Donor maps are fixed, not a random sample of features or donors.\n\nNaturalness is measured separately: support expansion beyond k=10, effective spatial positions, peak coordinate changes, lower-face pattern energy and post-reencoding discrepancy. Equal output L2 norms do not equate these properties. The localized control is the recipient's geometric lower-face mask; raw donor maps are transferred without pose alignment or warping.\n\nSee paired_comparisons.csv for per-donor and pooled estimates, per_image_results.csv for every evaluated image, and failures_and_missingness.csv for generation failures. Per-image classifier confidence vectors and landmarks remain in evaluations/. No demographic causal specificity or beard-control claim follows from these proxy metrics alone.\n'''
    (run/'results/research_summary.md').write_text(text)
    print(json.dumps(info))
if __name__=='__main__':main()
