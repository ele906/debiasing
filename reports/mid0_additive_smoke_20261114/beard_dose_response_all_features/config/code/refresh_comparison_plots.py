"""Compact shared-scale ten-feature comparisons from completed summary CSVs."""
import argparse,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import read
MEN=['indian_man','middle_eastern_man','east_asian_man','black_man','hispanic_man','white_man']
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();rows=read(run/'results/paired_summary.csv');features=sorted({int(r['feature_id']) for r in rows});plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
 if len(features)<2:return
 for method,lowrange in [('multiplicative',False),('multiplicative',True),('additive',False)]:
  fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
  for ax,group in zip(axes.flat,MEN):
   for f in features:
    rs=[r for r in rows if int(r['feature_id'])==f and r['prompt_group']==group and r['method']==method and r['metric']=='delta_current_face_pp' and r['subset']=='all' and r['mean']]
    if method=='multiplicative':
     if lowrange:rs=[r for r in rs if abs(float(r['strength_label'][6:]))<=7]
     rs.sort(key=lambda r:float(r['strength_label'][6:]));xx=[float(r['strength_label'][6:]) for r in rs]
    else:
     rs=[r for r in rs if r['strength_label'] in ['fixed_10','fixed_20','fixed_50']];rs.sort(key=lambda r:int(r['strength_label'].split('_')[1]));xx=[int(r['strength_label'].split('_')[1]) for r in rs]
    ax.plot(xx,[float(r['mean']) for r in rs],label=str(f),lw=1.2)
   if method=='multiplicative':
    if not lowrange:ax.set_xscale('symlog',linthresh=7);ax.axvspan(-100,-50,color='gray',alpha=.12);ax.axvspan(50,100,color='gray',alpha=.12)
    ticks=[-7,-5,0,1,5,7] if lowrange else [-100,-50,-7,0,1,7,50,100];ax.set_xticks(ticks);ax.set_xticklabels([str(x) for x in ticks],rotation=45,fontsize=8);ax.axvline(1,color='gray',ls=':',lw=.6)
   else:ax.set_xticks([10,20,50])
   ax.axhline(0,color='gray',lw=.7);ax.set_title(group.replace('_',' '));ax.grid(axis='y',alpha=.15)
  axes.flat[0].legend(ncol=2,fontsize=7)
  fig.suptitle('Ten selected features: paired predicted beard-area change (shared y scale)\n'+('20 seeds per curve; α<0 sign reversal, α=0 removal, α=1 unchanged' if method=='multiplicative' else 'Five paired seeds per curve; z′ = z + β')+'; these are not random controls',fontsize=11)
  fig.supxlabel('Latent multiplier α' if method=='multiplicative' else 'Additive offset β');fig.supylabel('Current-face normalized change (percentage points)');fig.tight_layout()
  name='ten_feature_'+method+('_low_range' if lowrange else '')
  old=run/'logs/initial_comparison_figures';old.mkdir(exist_ok=True)
  for ext in ['png','pdf']:
   dest=run/'figures'/(name+'.'+ext)
   if dest.exists() and not (old/dest.name).exists():shutil.copy2(dest,old/dest.name)
   fig.savefig(dest,dpi=200,bbox_inches='tight')
  plt.close(fig)
 page=run/'README.md';text=page.read_text();link='![Ten features, multipliers −7 through +7](figures/ten_feature_multiplicative_low_range.png)\n\n'
 if link not in text:text=text.replace('## Figures\n\n','## Figures\n\n'+link)
 page.write_text(text)
if __name__=='__main__':main()
