"""Six-row additive photo sheets plus jet overlays from saved SAE activations."""
import csv,datetime as dt,hashlib,json,os,shutil,sys,concurrent.futures as cf
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
from prepare import ROOT,BASE,SAE,read,write,sha
RACES=['indian','middle_eastern','east_asian','black','hispanic','white']
SOURCE=BASE/'ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv'
ARCHIVES=BASE/'additive_20261114_174738_954258/workers'

def main():
 run=BASE/('additive_gender_gallery_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 for d in ['figures','maps','config/code','results']:(run/d).mkdir(parents=True,exist_ok=True)
 rows=[r for r in read(SOURCE) if r['seed']=='30' and r['strength_label'] in ['fixed_10','fixed_20','fixed_50']];assert len(rows)==360
 features=sorted({int(r['feature_id']) for r in rows});assert len(features)==10
 def verify(r):
  archive=ARCHIVES/('feature_'+r['feature_id'])/r['activation_path'];assert sha(archive)==r['activation_sha256'];assert sha(r['image_path'])==r['image_sha256'];assert r['layer']=='unet.mid_block.attentions.0'
  with np.load(archive,allow_pickle=False) as d:
   ids=d['feature_ids'];ix=np.flatnonzero(ids==int(r['feature_id']));assert len(ix)==1;idx=int(ix[0]);assert str(d['layer'])=='unet.mid_block.attentions.0'
   assert d['original_0'].shape==(1,16,16,10) and not any(k.startswith('original_') and k!='original_0' for k in d.files)
   original=d['original_0'][0,:,:,idx].copy();modified=d['modified_0'][0,:,:,idx].copy();beta=float(d['additive_offset_0']);timestep=float(d['timestep_0']);assert beta==float(r['additive_offset'])
   assert np.allclose(modified,original+beta,rtol=0,atol=1e-5)
  return dict(row=r,archive=archive,original=original,modified=modified,beta=beta,timestep=timestep)
 with cf.ThreadPoolExecutor(max_workers=8) as pool:loaded=list(pool.map(verify,rows))
 lookup={(int(v['row']['feature_id']),v['row']['group'],int(v['beta'])):v for v in loaded};panels=[];scales=[];allmaps={}
 for f in features:
  maximum=0.
  for gender in ['man','woman']:
   for race in RACES:
    group=race+'_'+gender;first=lookup[f,group,10]
    for beta in [0,10,20,50]:
     v=first if beta==0 else lookup[f,group,beta];assert np.array_equal(v['original'],first['original']);values=v['original'] if beta==0 else v['modified'];assert values.shape==(16,16) and np.isfinite(values).all()
     key=f'feature_{f}_{group}_seed030_beta{beta}';path=run/'maps'/(key+'.npz');np.savez_compressed(path,original_feature_map=v['original'],modified_feature_map=values,feature_id=np.array(f),additive_offset=np.array(beta,dtype=np.float32),timestep=np.array(v['timestep']),layer=np.array('unet.mid_block.attentions.0'))
     image=Path(v['row']['baseline_image_path'] if beta==0 else v['row']['image_path']);imagehash=sha(image)
     with Image.open(image) as im:assert im.size==(512,512)
     allmaps[f,group,beta]=dict(values=values,image=image)
     maximum=max(maximum,float(np.abs(values).max()))
     panels.append(dict(feature_id=f,prompt_group=group,seed=30,additive_offset=beta,source_image=str(image),image_sha256=imagehash,source_activation_archive=str(v['archive']),source_activation_sha256=v['row']['activation_sha256'],map_path=str(path.relative_to(run)),map_sha256=sha(path),map_origin='paired pre-edit coordinate' if beta==0 else 'modified SAE coordinate',activation_mean=float(values.mean()),activation_max=float(values.max()),activation_nonzero_fraction=float((values!=0).mean()),layer='unet.mid_block.attentions.0',timestep=v['timestep']))
  scales.append(dict(feature_id=f,vmin=0,vmax=maximum,cmap='jet',quantity='absolute SAE coordinate activation',scope='All12 prompt groups and baseline/+10/+20/+50 for this feature',interpolation='nearest from16x16 to512x512, then thumbnail',opacity=.45,zero_activation='transparent'))
 font=lambda size:ImageFont.truetype('DejaVuSans.ttf',size)
 jet=colormaps['jet'];index=[]
 for scale in scales:
  f=scale['feature_id'];vmax=scale['vmax'] or 1.
  for gender in ['man','woman']:
   label='men' if gender=='man' else 'women'
   for overlay in [False,True]:
    tile=256;top=90;rowh=292;footer=100 if overlay else 25;width=1024;height=top+6*rowh+footer;canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    title=f'Feature {f}: additive interventions ({label})'+(' — activation overlay' if overlay else '')
    draw.text((12,8),title,font=font(22),fill='black');draw.text((12,38),'Seed30',font=font(15),fill='#444444')
    for j,title in enumerate(['Baseline','Add +10','Add +20','Add +50']):draw.text((j*tile+12,63),title,font=font(17),fill='black')
    for i,race in enumerate(RACES):
     group=race+'_'+gender;y=top+i*rowh;draw.text((12,y+2),race.replace('_',' ').title(),font=font(17),fill='black')
     for j,beta in enumerate([0,10,20,50]):
      v=allmaps[f,group,beta]
      with Image.open(v['image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
      if overlay:
       magnitude=np.abs(v['values']);expanded=np.repeat(np.repeat(magnitude,32,axis=0),32,axis=1);color=np.asarray(jet(expanded/vmax))[...,:3]*255.;alpha=np.where(expanded>0,.45,0)[...,None];rgb=np.rint(rgb*(1-alpha)+color*alpha).astype(np.uint8)
      preview=Image.fromarray(rgb).resize((tile,tile),Image.Resampling.LANCZOS);canvas.paste(preview,(j*tile,y+29))
    if overlay:
     y=top+6*rowh+8;draw.text((12,y),'Activation magnitude',font=font(15),fill='black');gradient=np.asarray(jet(np.linspace(0,1,700)))[None,:,:3];gradient=np.repeat(np.uint8(np.rint(gradient*255)),22,axis=0);canvas.paste(Image.fromarray(gradient),(250,y))
     for pos in [0,.25,.5,.75,1]:draw.text((250+int(700*pos)-9,y+27),f'{vmax*pos:.1f}',font=font(14),fill='black')
     draw.text((12,y+58),'One scale per feature; zero activation is transparent.',font=font(14),fill='#444444')
    name=f'feature_{f}_{label}'+('_jet' if overlay else '')+'.png';path=run/'figures'/name;canvas.save(path)
    index.append(dict(feature_id=f,prompt_gender=label,variant='jet_overlay' if overlay else 'photos',path='figures/'+name,rows=6,columns=4,seed=30,vmin=0 if overlay else '',vmax=vmax if overlay else ''))
  print('Sheets ready',f,flush=True)
 write(run/'results/panels.csv',panels);write(run/'results/color_scales.csv',scales);write(run/'results/figure_index.csv',index)
 cfg=dict(run_id=run.name,source=str(SOURCE),source_sha256=sha(SOURCE),feature_ids=features,seed=30,races=RACES,offsets=[0,10,20,50],sae_sha256=SAE,layer='unet.mid_block.attentions.0',n_panels=len(panels),n_photo_sheets=20,n_overlay_sheets=20,n_generated_images=0,n_models_run=0,color_scale='Shared within each feature across both genders and all conditions; scales differ between features',overlay_source='Modified selected SAE coordinate; baseline uses original map from paired pre-edit archive',numpy_version=np.__version__)
 (run/'config/experiment_config.json').write_text(json.dumps(cfg,indent=2));shutil.copy2(__file__,run/'config/code/additive_gallery.py');shutil.copy2(Path(__file__).parent/'prepare.py',run/'config/code/prepare.py')
 text='# Additive interventions: men and women\n\nEach sheet has six race-prompt rows and four columns: baseline, add+10, add+20, add+50. All images are saved seed30 generations, matching the earlier gallery. No diffusion generation or new model inference was performed. Original512×512 PNGs remain untouched; these are256px viewing derivatives.\n\nJet overlays show the magnitude of the selected **modified SAE coordinate**, not beard segmentation or causal attribution. Maps are16×16, enlarged using nearest-neighbor interpolation before thumbnailing. Zero activation is transparent; positive activation uses45% overlay opacity. Each feature has one shared scale across both genders and all four columns, with no per-image normalization or percentile clipping. Scales differ between features: read each colorbar before comparing colors.\n\nAdding β at every spatial position can make the overlays broad even where natural activation was sparse. The baseline map comes from the paired pre-edit activation archive; its equality across the three offsets was verified. The heatmaps show latents before casting the decoded intervention to the model output dtype.\n\n| Feature | Men | Women | Men + jet | Women + jet |\n|---|---|---|---|---|\n'
 for f in features:text+=f'| {f} | [Photos](figures/feature_{f}_men.png) | [Photos](figures/feature_{f}_women.png) | [Overlay](figures/feature_{f}_men_jet.png) | [Overlay](figures/feature_{f}_women_jet.png) |\n'
 text+='\n## Reproduce\n\n```bash\nPYTHONNOUSERSITE=1 PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python config/code/additive_gallery.py\n```\n\nThe script creates a new timestamped directory. config records input identity and code; results/panels.csv links each tile to its unchanged source image and activation archive, with hashes. maps/ retains the selected original and modified float32 feature maps losslessly. The original full activation archives remain at their recorded paths.\n'
 (run/'README.md').write_text(text)
 write(run/'output_manifest.csv',[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.csv'])
 print('RUN_DIRECTORY',run,flush=True)
if __name__=='__main__':main()
