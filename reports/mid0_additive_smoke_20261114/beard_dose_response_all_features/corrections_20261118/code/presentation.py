"""Plot-only presentation revision: short titles and four panels for each prompt gender."""
import argparse,datetime as dt,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import read,write,sha
MEN=['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man']

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output-dir',type=Path);a=p.parse_args();source=a.source_run.resolve()
 run=a.output_dir or source.parent/('beard_plot_presentation_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 if run.exists():raise FileExistsError('Refusing to overwrite existing figure revision')
 for d in ['config/code','results','figures']:(run/d).mkdir(parents=True,exist_ok=True)
 inputs=[source/'results/per_image_scores.csv',source/'results/paired_summary.csv'];rows=[r for r in read(inputs[0]) if r['feature_id']=='4228'];summary=[r for r in read(inputs[1]) if r['feature_id']=='4228' and r['metric']=='delta_current_face_pp']
 write(run/'config/input_hashes.csv',[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in inputs])
 shutil.copy2(__file__,run/'config/code/presentation.py');shutil.copy2(Path(__file__).parent/'prepare.py',run/'config/code/prepare.py')
 plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42})
 titles={'all':'Feature 4228: beard area change (all seeds)','baseline_bearded':'Feature 4228: beard area change (baseline-bearded)','additive':'Feature 4228: beard area change (additive)','norm':'Feature 4228: beard area change vs intervention size'}
 produced=[];samples=[]
 for gender in ['men','women']:
  groups=MEN if gender=='men' else [g[:-3]+'woman' for g in MEN]
  for kind in ['all','baseline_bearded','additive','norm']:
   fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
   for ax,group in zip(axes.flat,groups):
    rs=[r for r in rows if r['prompt_group']==group and r['delta_current_face_pp']!='']
    if kind in ['all','baseline_bearded']:
     rs=[r for r in rs if r['method']=='multiplicative' and (kind=='all' or r['baseline_bearded_proxy']=='True')]
     for seed in sorted({r['seed'] for r in rs}):
      ss=sorted([r for r in rs if r['seed']==seed],key=lambda r:float(r['multiplier']));ax.plot([float(r['multiplier']) for r in ss],[float(r['delta_current_face_pp']) for r in ss],color='#4477AA',lw=.7,alpha=.2)
     ss=sorted([r for r in summary if r['method']=='multiplicative' and r['prompt_group']==group and r['subset']==kind and r['mean']!=''],key=lambda r:float(r['strength_label'][6:]));xx=[float(r['strength_label'][6:]) for r in ss]
     ax.plot(xx,[float(r['mean']) for r in ss],color='#224488',lw=1.8);ax.fill_between(xx,[float(r['ci95_low']) if r['ci95_low'] else np.nan for r in ss],[float(r['ci95_high']) if r['ci95_high'] else np.nan for r in ss],color='#4477AA',alpha=.2)
     ax.set_xscale('symlog',linthresh=7);ticks=[-100,-50,-7,0,1,7,50,100];ax.set_xticks(ticks);ax.set_xticklabels([str(x) for x in ticks],rotation=45,fontsize=8)
     ax.axvline(1,color='#228833',ls=':',lw=.8);ax.axvspan(-100,-50,color='gray',alpha=.1);ax.axvspan(50,100,color='gray',alpha=.1)
     if not rs:ax.text(.5,.55,'No eligible baseline-bearded seeds',transform=ax.transAxes,ha='center',fontsize=9)
    elif kind=='additive':
     rs=[r for r in rs if r['method']=='additive' and r['strength_label'] in ['fixed_10','fixed_20','fixed_50']]
     for seed in sorted({r['seed'] for r in rs}):
      ss=sorted([r for r in rs if r['seed']==seed],key=lambda r:float(r['additive_offset']));ax.plot([float(r['additive_offset']) for r in ss],[float(r['delta_current_face_pp']) for r in ss],color='#CC6677',lw=.8,alpha=.35)
     ss=[next(r for r in summary if r['method']=='additive' and r['prompt_group']==group and r['subset']=='all' and r['strength_label']==f'fixed_{off}') for off in [10,20,50]]
     ax.plot([10,20,50],[float(r['mean']) for r in ss],'-o',color='#992244');ax.fill_between([10,20,50],[float(r['ci95_low']) for r in ss],[float(r['ci95_high']) for r in ss],color='#CC6677',alpha=.2);ax.set_xticks([10,20,50])
    else:
     rs=[r for r in rs if r['method'] in ['multiplicative','additive'] and r['applied_l2']!='']
     branches=[('Multiplier α ≥ 0','#4477AA','o',lambda r:r['method']=='multiplicative' and float(r['multiplier'])>=0),('Multiplier α < 0','#AA3377','x',lambda r:r['method']=='multiplicative' and float(r['multiplier'])<0),('Additive','#228833','^',lambda r:r['method']=='additive')]
     for label,color,marker,select in branches:
      ss=[r for r in rs if select(r)];ax.scatter([float(r['applied_l2']) for r in ss],[float(r['delta_current_face_pp']) for r in ss],s=12,alpha=.35,c=color,marker=marker,label=label)
     ax.set_xscale('symlog',linthresh=1);ax.legend(fontsize=8,frameon=False)
    ax.axhline(0,color='gray',lw=.7);ax.set_title(group.replace('_',' ').title(),fontsize=11);ax.grid(axis='y',alpha=.12)
    if rs and max(abs(float(r['delta_current_face_pp'])) for r in rs)==0:ax.set_ylim(-.05,.05)
    samples.append(dict(gender=gender,graph=kind,prompt_group=group,n_conditions=len(rs),n_seed_clusters=len({r['seed'] for r in rs})))
   fig.suptitle(titles[kind],fontsize=16)
   fig.supxlabel('Multiplier α' if kind in ['all','baseline_bearded'] else 'Additive offset β' if kind=='additive' else 'Applied intervention size (L2)',fontsize=12)
   fig.supylabel('Area change (%)',fontsize=12)
   suffix=f'{gender}_{kind}_delta_current_face_pp' if kind in ['all','baseline_bearded'] else f'{gender}_additive_fixed' if kind=='additive' else f'{gender}_applied_norm'
   name='feature4228_'+suffix
   for ext in ['png','pdf']:fig.savefig(run/'figures'/(name+'.'+ext),dpi=220)
   # Preserve existing direct men-only additive/norm links with the updated style.
   if gender=='men' and kind in ['additive','norm']:
    alias='feature4228_additive_fixed' if kind=='additive' else 'feature4228_applied_norm'
    for ext in ['png','pdf']:shutil.copy2(run/'figures'/(name+'.'+ext),run/'figures'/(alias+'.'+ext))
   produced.append(dict(gender=gender,kind=kind,title=titles[kind],path='figures/'+name+'.png'));plt.close(fig)
 write(run/'results/panel_sample_sizes.csv',samples);write(run/'results/figure_index.csv',produced)
 (run/'results/complete.json').write_text(json.dumps(dict(n_new_generated_images=0,n_models_run=0,n_trained_models=0,n_main_figures=8,source_run=str(source),purpose='Presentation only; existing data and summaries unchanged'),indent=2))
 text='# Feature 4228: beard area changes\n\nFour views of the existing experiment, shown separately for men’s and women’s prompt groups. This revision changes presentation only; no images were generated or rescored.\n\n'
 for gender in ['men','women']:
  text+=f'## {gender.title()}’s prompt groups\n\n'
  for i,r in enumerate([r for r in produced if r['gender']==gender],1):text+=f'### {i}. {r["title"].split(": ",1)[1]}\n\n![{r["title"]}]({r["path"]})\n\n'
 text+='## How to read the figures\n\nBeard area is the parser’s class14 area divided by its facial classes1–11 and14, excluding neck and hair. The axis label “Area change (%)” means edited beard-area percentage minus the same-prompt, same-seed control percentage:10% to12% is shown as+2. This is a percentage-point difference, not relative percentage growth. These remain parser measurements rather than validated beard masks. Prompt gender does not establish facial-hair presence.\n\n1. **All seeds:**20 multiplicative seeds were evaluated per group. Each point averages available normalized scores; extreme interventions can have fewer because predicted face area is zero. Exact counts are in paired_summary.csv. Thin lines are individual seeds, the bold line is their mean, and shading around it is the existing95% seed-bootstrap interval. α=0 removes the coordinate, α=1 leaves it unchanged, α<0 reverses sign, and α>1 amplifies. Gray end bands mark |α|≥50 as a caution range, not a per-image distortion verdict.\n2. **Baseline-bearded:** the same analysis restricted to baseline parsed beard/face fraction≥1%. A panel with no eligible seeds is missing a qualifying sample, not a measured zero effect.\n3. **Additive:** five seeds per group at offsets10,20,50. Each edit is compared with its saved no-intervention baseline.\n4. **Applied L2:** L2 is the magnitude of the actual block-output change after dtype casting: `sqrt(sum((edited_block - original_block)**2))`. It is a measurement, **not regularization or a training penalty**. For a change vector(3,4), L2=5. Opposite changes(3,4) and(−3,−4) have the same L2, so size alone does not describe direction. The different markers identify intervention methods/signs. This scatter plot helps inspect responses at comparable sizes; it does not by itself establish a direction effect after controlling for magnitude.\n\nPanel y scales vary so small responses remain visible. Graph4 contains20 multiplicative seeds and five additive seeds; graph3 is not a magnitude-matched comparison with graph1. No new uncertainty calculations or statistical tests are introduced.\n\n## Data and reproduction\n\nThe original scored tables, archives, checkpoints, input hashes and methodological details are preserved in the [original scoring report](../beard_dose_response_4228_original/README.md) and the local source run recorded in config/input_hashes.csv. The unchanged scored tables remain in this report’s results/ directory on GitHub.\n\n```bash\nPYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python config/code/presentation.py --source-run '+str(source)+' --output-dir /path/to/new_figure_directory\n```\n'
 # Existing original README is retained under a stable file in the same report.
 text=text.replace('../beard_dose_response_4228_original/README.md','ORIGINAL_ANALYSIS.md')
 (run/'README.md').write_text(text);shutil.copy2(source/'README.md',run/'ORIGINAL_ANALYSIS.md')
 write(run/'output_manifest.csv',[dict(path=str(f.relative_to(run)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='output_manifest.csv'])
 print('RUN_DIRECTORY',run,flush=True);print('Created8 figures for both prompt genders; original data and figures preserved.',flush=True)
if __name__=='__main__':main()
