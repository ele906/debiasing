"""Two fixed nuisance-model sensitivities plus same-support paired comparisons. Existing tables only."""
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import importlib.util
entry=Path(__file__).with_name('analysis.py')
if not entry.exists():entry=Path(__file__).with_name('analyze.py')
spec=importlib.util.spec_from_file_location('bounded_analysis',entry)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

PRIMARY=('post_reencoded_discrepancy','pixel_mae','lowerface_change_fraction')

def load(run):
    rows=list(csv.DictReader((run/'results/per_image_analysis.csv').open()))
    for r in rows:
        for k in ['seed','target_norm','actual_applied_norm','support_expansion','balance_weight','original_activation_nonzero_fraction',*a.OUTCOMES]:
            r[k]=a.number(r.get(k))
        r['seed']=int(r['seed']);r['shuffle_seed']=int(r['shuffle_seed']) if r.get('shuffle_seed') else None
    return rows


def blocked_design(rows,spec):
    if spec=='adjusted_without_ppm':
        D,names=original_design(rows,'adjusted');keep=[i for i,n in enumerate(names) if n!='applied_norm_deviation_ppm']
        return D[:,keep],[names[i] for i in keep]
    if spec=='paired_strength_blocks':
        n=len(rows);cols=[np.ones(n)];names=['intercept']
        # Recipient x nominal magnitude blocks; method, donor and shuffle effects can differ by target.
        levels=sorted({(r['prompt_id'],r['target_norm']) for r in rows})
        for level in levels[1:]:cols.append(np.array([(r['prompt_id'],r['target_norm'])==level for r in rows],float));names.append('recipient_target='+str(level))
        targets=sorted({r['target_norm'] for r in rows})
        methods=sorted({r['method'] for r in rows})
        for target in targets:
            for method in methods[1:]:cols.append(np.array([r['method']==method and r['target_norm']==target for r in rows],float));names.append('method_target='+method+str(target))
            cols.append(np.array([(-.5 if r['donor_id']=='black_man_seed030' else .5 if r['donor_id']=='black_man_seed031' else 0)*(r['target_norm']==target) for r in rows]));names.append('donor_target='+str(target))
            for seed in (1729,2718):
                cols.append(np.array([((float(r['shuffle_seed']==seed)-1/3) if r['method']=='shuffled' else 0)*(r['target_norm']==target) for r in rows]));names.append('shuffle_target='+str(seed)+str(target))
        return np.column_stack(cols),names
    return original_design(rows,spec)


def main():
    run=Path(sys.argv[1]).resolve();rows=load(run)
    a.dump(run/'config/sensitivity_config.json',dict(specs=['adjusted_without_ppm','paired_strength_blocks'],
        rationale='Check numerical applied-norm jitter control and allow response effects to vary by target within paired blocks; no outcome-selected model search',
        same_support_rule='Natural compared with all three shuffled replicas only if their support equals natural exactly; donor-average inside recipient seed',
        inactive_rule='Original selected-feature map nonzero fraction exactly zero'))
    a.design=blocked_design
    results=[];diagnostics=[]
    for spec in ('adjusted_without_ppm','paired_strength_blocks'):
        r,d=a.regression(rows,spec,'all_methods');results.extend(r);diagnostics.extend(d)
    inactive=[r for r in rows if r['original_activation_nonzero_fraction']==0]
    for method in ('natural','shuffled','uniform','localized'):
        subset=[r for r in inactive if r['method']==method]
        r,d=a.regression(subset,'adjusted','inactive_'+method);results.extend(r);diagnostics.extend(d)
    a.write_csv(run/'results/model_sensitivity.csv',results);a.write_csv(run/'results/model_sensitivity_diagnostics.csv',diagnostics)
    pairs=[];summaries=[]
    for group,target,metric in itertools.product(a.GROUPS,sorted({r['target_norm'] for r in rows}),PRIMARY):
        values=[]
        for seed in range(30,35):
            differences=[];donor_count=0
            for donor in ('black_man_seed030','black_man_seed031'):
                natural=[r for r in rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']=='natural' and r['donor_id']==donor]
                shuffled=[r for r in rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']=='shuffled' and r['donor_id']==donor]
                if len(natural)!=1 or len(shuffled)!=3:continue
                support=natural[0]['support_expansion']
                if not all(r['support_expansion']==support for r in shuffled):continue
                d=natural[0][metric]-np.mean([r[metric] for r in shuffled])
                if np.isfinite(d):differences.append(float(d));donor_count+=1
            if donor_count==2:
                value=float(np.mean(differences));values.append(value)
                pairs.append(dict(group=group,target_norm=target,outcome=metric,seed=seed,n_donors=2,difference=value,support_difference=0.))
        v=np.array(values)
        if len(v):
            ix=np.array(list(itertools.product(range(len(v)),repeat=len(v))));boot=v[ix].mean(1)
            summaries.append(dict(group=group,target_norm=target,outcome=metric,mean_difference=float(v.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                n_seed_clusters=len(v),n_ineligible_seed_clusters=5-len(v),contrast='natural_minus_mean3_shuffles_at_identical_support'))
    a.write_csv(run/'results/same_support_paired_differences.csv',summaries);a.write_csv(run/'results/same_support_seed_differences.csv',pairs)
    print('Sensitivity models and same-support comparisons complete')
    for r in results:
        if r['spec']=='paired_strength_blocks' and r['scope']=='all_methods' and r['outcome'] in PRIMARY:
            print(r['outcome'],r.get('slope_per_1pp'),r.get('ci_low'),r.get('ci_high'))

original_design=a.design
if __name__=='__main__':main()
