"""Concise data-derived interpretation and integrity checks; no new inference."""
import argparse,csv,json
from pathlib import Path
from prepare import read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();rows=read(run/'results/per_image_scores.csv');summary=read(run/'results/paired_summary.csv');init=read(run/'results/additive_initiation_summary.csv')
 checks=[]
 for r in rows:
  assert sum(int(r[f'class_{i}_pixels']) for i in range(19))==262144
  assert int(r['beard_pixels'])==int(r['class_14_pixels'])
  if r['method']=='baseline' or (r['method']=='multiplicative' and float(r['multiplier'])==1):
   assert float(r['delta_image_pp'])==0
   if r['delta_current_face_pp']:assert float(r['delta_current_face_pp'])==0
  if r['beard_fraction_current_face']:assert 0<=float(r['beard_fraction_current_face'])<=1
 cfg=json.loads((run/'config/experiment_config.json').read_text());assert len(rows)==cfg['n_conditions'];controls=read(run/'config/control_equivalence.csv')
 result=dict(n_conditions=len(rows),method_counts={m:sum(r['method']==m for r in rows) for m in ['multiplicative','additive','baseline']},all_class_counts_conserved=True,all_noop_deltas_zero=True,normalization_bounds_checked=True,n_pixel_equal_controls=sum(r['pixels_equal']=='True' for r in controls),n_control_checks=len(controls))
 (run/'results/integrity_checks.json').write_text(json.dumps(result,indent=2))
 selected=[r for r in summary if r['feature_id']=='4228' and r['metric']=='delta_current_face_pp' and r['subset']=='all' and r['prompt_group'] in ['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man'] and ((r['method']=='multiplicative' and r['strength_label'] in ['alpha_-7','alpha_0','alpha_7']) or (r['method']=='additive' and r['strength_label']=='fixed_50'))]
 write(run/'results/feature4228_selected_effects.csv',selected)
 additive=[r for r in rows if r['method']=='additive'];initiation_rows=[];candidate_rows=[]
 for f in sorted({r['feature_id'] for r in additive},key=int):
  for subset in ['baseline_parser_zero','baseline_below_one_percent']:
   rr=[r for r in additive if r['feature_id']==f and r['baseline_beard_fraction_face'] and (int(r['baseline_beard_pixels'])==0 if subset=='baseline_parser_zero' else float(r['baseline_beard_fraction_face'])<.01)]
   contexts={(r['prompt_group'],r['seed']) for r in rr};cross=[r for r in rr if r['beard_fraction_current_face'] and float(r['beard_fraction_current_face'])>=.01];crosscontexts={(r['prompt_group'],r['seed']) for r in cross}
   inactivecontexts={(r['prompt_group'],r['seed']) for r in cross if r['original_activation_zero']=='True'}
   initiation_rows.append(dict(feature_id=f,subset=subset,edited_fraction_threshold=.01,n_unique_eligible_contexts=len(contexts),n_contexts_crossing_at_any_existing_offset=len(crosscontexts),n_inactive_contexts_crossing_at_any_offset=len(inactivecontexts),n_eligible_seed_clusters=len({s for _,s in contexts}),n_crossing_seed_clusters=len({s for _,s in crosscontexts}),interpretation='Parser threshold crossings, not verified new beards; repeated offsets count once per context'))
   for r in cross:candidate_rows.append(dict(feature_id=f,subset=subset,condition_id=r['condition_id'],prompt_group=r['prompt_group'],seed=r['seed'],strength_label=r['strength_label'],additive_offset=r['additive_offset'],applied_l2=r['applied_l2'],original_activation_zero=r['original_activation_zero'],baseline_beard_pixels=r['baseline_beard_pixels'],edited_beard_pixels=r['beard_pixels'],baseline_fraction=r['baseline_beard_fraction_face'],edited_fraction=r['beard_fraction_current_face'],image_path=r['image_path'],mask_path=r['beard_mask_path']))
 write(run/'results/initiation_unique_contexts.csv',initiation_rows)
 write(run/'results/initiation_candidates.csv',candidate_rows or [dict(status='no_one_percent_crossings')])
 text='## Measured results for feature4228\n\nThese are paired changes in the new parser’s predicted beard fraction of its facial classes, in percentage points. The multiplicative rows use20 seeds and the additive rows use5; their perturbation magnitudes are not matched by this table. Intervals resample seeds and do not include parser error.\n\n| Prompt group | Intervention | Seeds | Mean change (pp) | Exploratory95% interval |\n|---|---|---:|---:|---|\n'
 order=['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man']
 for group in order:
  for label in ['alpha_-7','alpha_0','alpha_7','fixed_50']:
   r=next(r for r in selected if r['prompt_group']==group and r['strength_label']==label);text+=f"| {group} | {label} | {r['n_seed_clusters']} | {float(r['mean']):+.2f} | [{float(r['ci95_low']):+.2f}, {float(r['ci95_high']):+.2f}] |\n"
 text+='\nThe parser responds to both multiplicative and additive interventions in some contexts. Therefore “additive features do nothing” is not supported by these measurements. Removing feature4228 (α=0) and reversing its sign (α<0) are distinct interventions. Low response in another prompt group does not establish race as a cause: natural activation, perturbation size, baseline facial hair, other representations and parser behavior remain possible explanations.\n\n'
 crossings=[r for r in init if r['feature_id']=='4228' and r['strength_label']=='fixed_50' and float(r['threshold'])==.01]
 text+='For +50 addition to feature4228, using a provisional1% beard/face threshold, parser crossings among baseline-below-threshold contexts are: '+', '.join(f"{r['prompt_group']} {r['n_crossings']}/{r['n_eligible']}" for r in crossings)+'. A zero denominator here means no eligible seed contexts, not a negative result. Crossings are candidate cases for visual inspection, not validated new beards. Threshold sensitivity is retained in the CSVs.\n\n'
 text+='Matched-norm random-feature/direction controls were not found in the inspected face-intervention archives. Other selected features are descriptive comparisons, not random controls. No feature-specific or demographic causal conclusion is established.\n'
 (run/'FINDINGS.md').write_text(text)
 readme=(run/'README.md').read_text();heading='## Measured results for feature4228'
 if heading not in readme:readme=readme.replace('## Figures\n\n',text+'\n## Figures\n\n')
 (run/'README.md').write_text(readme);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
